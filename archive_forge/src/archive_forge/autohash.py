import hashlib

# File to calculate the hash of
file_path = "path/to/file"

# Create a SHA-256 hash object
sha256_hash = hashlib.sha256()

# Open the file and read its contents
with open(file_path, "rb") as file:
    for chunk in iter(lambda: file.read(4096), b""):
        sha256_hash.update(chunk)

# Get the hexadecimal representation of the hash
hash_hex = sha256_hash.hexdigest()

print(f"SHA-256 hash of the file: {hash_hex}")
