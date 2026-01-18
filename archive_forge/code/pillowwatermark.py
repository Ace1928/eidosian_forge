from PIL import Image, ImageDraw, ImageFont
import os

# Directory containing the images
image_directory = "path/to/image/directory"

# Watermark text and font
watermark_text = "Your Watermark"
font = ImageFont.truetype("arial.ttf", 36)

# Iterate over the images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Open the image
        image_path = os.path.join(image_directory, filename)
        image = Image.open(image_path)
    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Calculate the position of the watermark
    text_width, text_height = draw.textsize(watermark_text, font)
    x = image.width - text_width - 10
    y = image.height - text_height - 10

    # Draw the watermark on the image
    draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255, 128))

    # Save the watermarked image
    watermarked_filename = f"watermarked_{filename}"
    watermarked_path = os.path.join(image_directory, watermarked_filename)
    image.save(watermarked_path)

    print(f"Watermarked: {filename}")
