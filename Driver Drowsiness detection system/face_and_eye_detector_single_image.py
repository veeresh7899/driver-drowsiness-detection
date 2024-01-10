import cv2 as cv

# Load face cascade and eye cascade from haarcascades folder
face_cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascades/haarcascade_eye.xml")

# Read image and convert it to grayscale
img = cv.imread('images/VEERESH.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect all faces in the image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Draw rectangles around faces and detect eyes in faces
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Region of Interest (ROI) for the face
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    # Detect eyes in the face region
    eyes = eye_cascade.detectMultiScale(roi_gray)

    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

# Display the modified image
cv.imshow('Detected Faces and Eyes', img)
cv.waitKey(0)

# Save the modified image
cv.imwrite('images/result1.jpg', img)
# Close all OpenCV windows
cv.destroyAllWindows()
