from PIL import Image
import os

# Directory containing the images
image_directory = "path/to/image/directory"

# Iterate over the images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(".png") or filename.endswith(".bmp"):
        # Open the image
        image_path = os.path.join(image_directory, filename)
        image = Image.open(image_path)

        # Convert the image to JPEG format
        new_filename = os.path.splitext(filename)[0] + ".jpg"
        new_image_path = os.path.join(image_directory, new_filename)
        image.save(new_image_path, "JPEG")

        print(f"Converted: {filename} -> {new_filename}")

print("Image conversion completed.")
