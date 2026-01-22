import logging
from cryptography.fernet import Fernet

KEY_FILE = "encryption.key"


def get_valid_encryption_key() -> bytes:
    """
    Ensures the encryption key's validity or generates a new one.

    Tries to read the encryption key from a local file. If the key is not valid
    or the file does not exist, generates a new key and stores it in the file.

    Returns:
        bytes: The valid encryption key.
    """
    try:
        with open(KEY_FILE, "rb") as file:
            key = file.read()
            Fernet(
                key
            )  # This line validates the key by attempting to create a Fernet instance
            logging.debug("Encryption key loaded successfully.")
            return key
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Loading encryption key failed: {e}")
    # Key is either invalid or not found; generate a new one
    new_key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as file:
        file.write(new_key)
    logging.info("Generated and stored a new encryption key.")
    return new_key
