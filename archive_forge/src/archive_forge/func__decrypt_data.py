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
def _decrypt_data(self, data: bytes, timestamp: int, time_info: typing.Optional[typing.Tuple[int, int]]) -> bytes:
    if time_info is not None:
        ttl, current_time = time_info
        if timestamp + ttl < current_time:
            raise InvalidToken
        if current_time + _MAX_CLOCK_SKEW < timestamp:
            raise InvalidToken
    self._verify_signature(data)
    iv = data[9:25]
    ciphertext = data[25:-32]
    decryptor = Cipher(algorithms.AES(self._encryption_key), modes.CBC(iv)).decryptor()
    plaintext_padded = decryptor.update(ciphertext)
    try:
        plaintext_padded += decryptor.finalize()
    except ValueError:
        raise InvalidToken
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    unpadded = unpadder.update(plaintext_padded)
    try:
        unpadded += unpadder.finalize()
    except ValueError:
        raise InvalidToken
    return unpadded