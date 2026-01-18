import os

# Directory containing the files to rename
directory = "/path/to/directory"

# Prefix to add to the file names
prefix = "new_"

# Iterate over the files in the directory
for filename in os.listdir(directory):
    # Get the file extension
    _, extension = os.path.splitext(filename)

    # Create the new file name with the prefix
    new_filename = prefix + filename

    # Rename the file
    old_path = os.path.join(directory, filename)
    new_path = os.path.join(directory, new_filename)
    os.rename(old_path, new_path)

    print(f"Renamed: {filename} -> {new_filename}")
