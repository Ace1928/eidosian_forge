import pandas as pd
from cryptography.fernet import Fernet

# Load the dataset
data = pd.read_csv("dataset.csv")

# Generate a random encryption key
key = Fernet.generate_key()
fernet = Fernet(key)

# Define the columns to encrypt
columns_to_encrypt = ["ssn", "credit_card"]

# Encrypt the specified columns
for column in columns_to_encrypt:
    data[column] = data[column].apply(lambda x: fernet.encrypt(str(x).encode()))

# Save the encrypted dataset
data.to_csv("encrypted_dataset.csv", index=False)

# Save the encryption key
with open("encryption_key.key", "wb") as key_file:
    key_file.write(key)
