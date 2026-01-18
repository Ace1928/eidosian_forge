import bz2

# File to compress
file_to_compress = "example.txt"

# Compress the file
with open(file_to_compress, "rb") as file_in:
    with bz2.open(file_to_compress + ".bz2", "wb") as file_out:
        file_out.write(file_in.read())

print("File compressed successfully.")

# Decompress the file
with bz2.open(file_to_compress + ".bz2", "rb") as file_in:
    with open("decompressed_file.txt", "wb") as file_out:
        file_out.write(file_in.read())

print("File decompressed successfully.")
