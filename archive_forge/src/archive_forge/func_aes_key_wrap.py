from __future__ import annotations
import typing
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers.algorithms import AES
from cryptography.hazmat.primitives.ciphers.modes import ECB
from cryptography.hazmat.primitives.constant_time import bytes_eq
def aes_key_wrap(wrapping_key: bytes, key_to_wrap: bytes, backend: typing.Any=None) -> bytes:
    if len(wrapping_key) not in [16, 24, 32]:
        raise ValueError('The wrapping key must be a valid AES key length')
    if len(key_to_wrap) < 16:
        raise ValueError('The key to wrap must be at least 16 bytes')
    if len(key_to_wrap) % 8 != 0:
        raise ValueError('The key to wrap must be a multiple of 8 bytes')
    a = b'\xa6\xa6\xa6\xa6\xa6\xa6\xa6\xa6'
    r = [key_to_wrap[i:i + 8] for i in range(0, len(key_to_wrap), 8)]
    return _wrap_core(wrapping_key, a, r)