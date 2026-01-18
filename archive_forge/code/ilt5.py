from PIL import Image
import os

# Directory containing the images
image_directory = "path/to/image/directory"

# Desired size for resized images
new_size = (800, 600)

# Iterate over the images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Open the image
        image_path = os.path.join(image_directory, filename)
        image = Image.open(image_path)

        # Resize the image
        resized_image = image.resize(new_size)

        # Save the resized image
        resized_image.save(os.path.join(image_directory, "resized_" + filename))

print("Images resized successfully.")
