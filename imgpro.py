import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_classificaton(ratio):
    ratio = round(ratio, 1)
    toret = ""
    if ratio >= 3:
        toret = "Slender"
    elif ratio >= 2.1 and ratio < 3:
        toret = "Medium"
    elif ratio >= 1.1 and ratio < 2.1:
        toret = "Bold"
    elif ratio <= 1:
        toret = "Round"
    toret = "(" + toret + ")"
    return toret

print("Starting")

# Load image in grayscale
img = cv2.imread('rice.png', 0)
if img is None:
    print("Error: rice.png not found!")
    exit()

# Convert into binary
ret, binary = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)

# Averaging filter
kernel = np.ones((5,5), np.float32)/9
dst = cv2.filter2D(binary, -1, kernel)

kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# Erosion
erosion = cv2.erode(dst, kernel2, iterations=1)

# Dilation
dilation = cv2.dilate(erosion, kernel2, iterations=1)

# Original edge detection (Canny on dilated)
edges = cv2.Canny(dilation, 100, 200)

# Canny on original grayscale
edges_canny = cv2.Canny(img, 100, 200)

# Sobel edge detection
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
edges_sobel = cv2.magnitude(sobelx, sobely)
edges_sobel = np.uint8(edges_sobel)

# Laplacian edge detection
edges_laplacian = cv2.Laplacian(img, cv2.CV_64F)
edges_laplacian = cv2.convertScaleAbs(edges_laplacian)

# Size detection and grain classification
contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("No. of rice grains =", len(contours))
total_ar = 0

# Count grain types

grain_types = {"Slender":0, "Medium":0, "Bold":0, "Round":0}

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    if aspect_ratio < 1:
        aspect_ratio = 1/aspect_ratio
    classification = get_classificaton(aspect_ratio).replace("(", "").replace(")", "")
    grain_types[classification] += 1
    print(round(aspect_ratio, 2), classification)
    total_ar += aspect_ratio

avg_ar = total_ar / len(contours)
avg_classification = get_classificaton(avg_ar).replace("(", "").replace(")", "")
print("Average Aspect Ratio =", round(avg_ar, 2), avg_classification)

# Print grain type counts
print("\nGrain type counts:")
for key, value in grain_types.items():
    print(f"{key}: {value}")

# Plot all images
plt.figure(figsize=(15,8))

plt.subplot(2,4,1)
plt.imshow(img, cmap='gray')
plt.title("Original image")
plt.axis('off')

plt.subplot(2,4,2)
plt.imshow(binary, cmap='gray')
plt.title("Binary image")
plt.axis('off')

plt.subplot(2,4,3)
plt.imshow(dst, cmap='gray')
plt.title("Filtered image")
plt.axis('off')

plt.subplot(2,4,4)
plt.imshow(erosion, cmap='gray')
plt.title("Eroded image")
plt.axis('off')

plt.subplot(2,4,5)
plt.imshow(dilation, cmap='gray')
plt.title("Dilated image")
plt.axis('off')

plt.subplot(2,4,6)
plt.imshow(edges, cmap='gray')
plt.title("Canny (dilated)")
plt.axis('off')

plt.subplot(2,4,7)
plt.imshow(edges_sobel, cmap='gray')
plt.title("Sobel edges")
plt.axis('off')

plt.subplot(2,4,8)
plt.imshow(edges_laplacian, cmap='gray')
plt.title("Laplacian edges")
plt.axis('off')

plt.tight_layout()
plt.show()

# plot grain type counts

plt.figure(figsize=(6,4))
plt.bar(grain_types.keys(), grain_types.values(), color=['green','orange','blue','red'])
plt.title("Number of Rice Grains by Type")
plt.xlabel("Grain Type")
plt.ylabel("Count")
plt.show()
    

