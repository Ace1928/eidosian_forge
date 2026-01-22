import subprocess

# Source directory
source_dir = "/path/to/source/directory/"

# Destination directory
dest_dir = "/path/to/destination/directory/"

# rsync command
rsync_command = ["rsync", "-avz", "--delete", source_dir, dest_dir]

# Execute the rsync command
subprocess.run(rsync_command, check=True)

print("File synchronization completed.")
