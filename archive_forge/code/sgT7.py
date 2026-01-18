from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

# File to encrypt
file_to_encrypt = "sensitive_data.txt"

# Generate a random encryption key
key = get_random_bytes(16)

# Create an AES cipher object
cipher = AES.new(key, AES.MODE_EAX)

# Read the file contents
with open(file_to_encrypt, "rb") as file:
    plaintext = file.read()

# Encrypt the data
ciphertext, tag = cipher.encrypt_and_digest(plaintext)

# Write the encrypted data to a new file
with open("encrypted_file.bin", "wb") as file:
    [file.write(x) for x in (cipher.nonce, tag, ciphertext)]

print("File encrypted successfully.")

# Decrypt the file
with open("encrypted_file.bin", "rb") as file:
    nonce, tag, ciphertext = [file.read(x) for x in (16, 16, -1)]

cipher = AES.new(key, AES.MODE_EAX, nonce)
plaintext = cipher.decrypt_and_verify(ciphertext, tag)

# Write the decrypted data to a new file
with open("decrypted_file.txt", "wb") as file:
    file.write(plaintext)

print("File decrypted successfully.")
