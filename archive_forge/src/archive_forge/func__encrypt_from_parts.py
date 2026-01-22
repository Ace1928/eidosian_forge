from __future__ import annotations
import base64
import binascii
import os
import time
import typing
from cryptography import utils
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.hmac import HMAC
def _encrypt_from_parts(self, data: bytes, current_time: int, iv: bytes) -> bytes:
    utils._check_bytes('data', data)
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data) + padder.finalize()
    encryptor = Cipher(algorithms.AES(self._encryption_key), modes.CBC(iv)).encryptor()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    basic_parts = b'\x80' + current_time.to_bytes(length=8, byteorder='big') + iv + ciphertext
    h = HMAC(self._signing_key, hashes.SHA256())
    h.update(basic_parts)
    hmac = h.finalize()
    return base64.urlsafe_b64encode(basic_parts + hmac)