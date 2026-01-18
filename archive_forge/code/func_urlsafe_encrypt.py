import base64
import os
import random
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers import modes
from oslo_utils import encodeutils
def urlsafe_encrypt(key, plaintext, blocksize=16):
    """
    Encrypts plaintext. Resulting ciphertext will contain URL-safe characters.
    If plaintext is Unicode, encode it to UTF-8 before encryption.

    :param key: AES secret key
    :param plaintext: Input text to be encrypted
    :param blocksize: Non-zero integer multiple of AES blocksize in bytes (16)

    :returns: Resulting ciphertext
    """

    def pad(text):
        """
        Pads text to be encrypted
        """
        pad_length = blocksize - len(text) % blocksize
        pad = b''.join((bytes((random.SystemRandom().randint(1, 255),)) for i in range(pad_length - 1)))
        return text + b'\x00' + pad
    plaintext = encodeutils.to_utf8(plaintext)
    key = encodeutils.to_utf8(key)
    init_vector = os.urandom(16)
    backend = default_backend()
    cypher = Cipher(algorithms.AES(key), modes.CBC(init_vector), backend=backend)
    encryptor = cypher.encryptor()
    padded = encryptor.update(pad(plaintext)) + encryptor.finalize()
    encoded = base64.urlsafe_b64encode(init_vector + padded)
    encoded = encoded.decode('ascii')
    return encoded