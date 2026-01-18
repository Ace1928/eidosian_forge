import os
import re

# Directory containing the files to rename
directory = "path/to/directory"

# Regular expression pattern for matching file names
pattern = r"file_(\d+)\.txt"

# Iterate over the files in the directory
for filename in os.listdir(directory):
    # Check if the file name matches the pattern
    match = re.match(pattern, filename)
    if match:
        # Extract the matched group (number)
        number = match.group(1)

        # Create the new file name
        new_filename = f"newfile_{number}.txt"

        # Rename the file
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)

        print(f"Renamed: {filename} -> {new_filename}")
