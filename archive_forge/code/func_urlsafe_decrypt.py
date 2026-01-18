import base64
import os
import random
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers import modes
from oslo_utils import encodeutils
def urlsafe_decrypt(key, ciphertext):
    """
    Decrypts URL-safe base64 encoded ciphertext.
    On Python 3, the result is decoded from UTF-8.

    :param key: AES secret key
    :param ciphertext: The encrypted text to decrypt

    :returns: Resulting plaintext
    """
    ciphertext = encodeutils.to_utf8(ciphertext)
    key = encodeutils.to_utf8(key)
    ciphertext = base64.urlsafe_b64decode(ciphertext)
    backend = default_backend()
    cypher = Cipher(algorithms.AES(key), modes.CBC(ciphertext[:16]), backend=backend)
    decryptor = cypher.decryptor()
    padded = decryptor.update(ciphertext[16:]) + decryptor.finalize()
    text = padded[:padded.rfind(b'\x00')]
    text = text.decode('utf-8')
    return text