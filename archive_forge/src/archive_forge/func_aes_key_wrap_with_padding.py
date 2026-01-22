from __future__ import annotations
import typing
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers.algorithms import AES
from cryptography.hazmat.primitives.ciphers.modes import ECB
from cryptography.hazmat.primitives.constant_time import bytes_eq
def aes_key_wrap_with_padding(wrapping_key: bytes, key_to_wrap: bytes, backend: typing.Any=None) -> bytes:
    if len(wrapping_key) not in [16, 24, 32]:
        raise ValueError('The wrapping key must be a valid AES key length')
    aiv = b'\xa6YY\xa6' + len(key_to_wrap).to_bytes(length=4, byteorder='big')
    pad = (8 - len(key_to_wrap) % 8) % 8
    key_to_wrap = key_to_wrap + b'\x00' * pad
    if len(key_to_wrap) == 8:
        encryptor = Cipher(AES(wrapping_key), ECB()).encryptor()
        b = encryptor.update(aiv + key_to_wrap)
        assert encryptor.finalize() == b''
        return b
    else:
        r = [key_to_wrap[i:i + 8] for i in range(0, len(key_to_wrap), 8)]
        return _wrap_core(wrapping_key, aiv, r)