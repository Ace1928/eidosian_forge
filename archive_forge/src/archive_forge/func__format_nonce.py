from dissononce.cipher.cipher import Cipher
from dissononce.exceptions.decrypt import DecryptFailedException
import struct
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.exceptions import InvalidTag
@staticmethod
def _format_nonce(n):
    return b'\x00\x00\x00\x00' + struct.pack('<Q', n)