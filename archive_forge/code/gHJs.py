import gzip

# File to compress
file_to_compress = "example.txt"

# Compress the file
with open(file_to_compress, "rb") as file_in:
    with gzip.open(file_to_compress + ".gz", "wb") as file_out:
        file_out.writelines(file_in)

print("File compressed successfully.")

# Decompress the file
with gzip.open(file_to_compress + ".gz", "rb") as file_in:
    with open("decompressed_file.txt", "wb") as file_out:
        file_out.writelines(file_in)

print("File decompressed successfully.")
