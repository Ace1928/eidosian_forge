import os
import shutil

# Define the main folder path
main_folder = "/path/to/your/folder"
# Create subfolders for different file types
subfolders = {
    "Documents": [".pdf", ".doc", ".docx", ".txt"],
    "Images": [".jpg", ".jpeg", ".png", ".gif"],
    "Videos": [".mp4", ".avi", ".mov"],
    "Audio": [".mp3", ".wav", ".aac"],
    "Others": [],
}
# Iterate over files in the main folder
for filename in os.listdir(main_folder):
    file_path = os.path.join(main_folder, filename)

    # Check if it's a file (not a folder)
    if os.path.isfile(file_path):
        # Get the file extension
        _, extension = os.path.splitext(filename)

        # Determine the appropriate subfolder
        subfolder = "Others"
        for folder, extensions in subfolders.items():
            if extension.lower() in extensions:
                subfolder = folder
                break

        # Create the subfolder if it doesn't exist
        subfolder_path = os.path.join(main_folder, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)

        # Move the file to the appropriate subfolder
        destination_path = os.path.join(subfolder_path, filename)
        shutil.move(file_path, destination_path)

        print(f"Moved {filename} to {subfolder} folder.")
