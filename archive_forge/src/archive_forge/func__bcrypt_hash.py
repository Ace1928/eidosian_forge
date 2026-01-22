import re
import struct
from functools import reduce
from Cryptodome.Util.py3compat import (tobytes, bord, _copy_bytes, iter_range,
from Cryptodome.Hash import SHA1, SHA256, HMAC, CMAC, BLAKE2s
from Cryptodome.Util.strxor import strxor
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.number import size as bit_size, long_to_bytes, bytes_to_long
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
def _bcrypt_hash(password, cost, salt, constant, invert):
    from Cryptodome.Cipher import _EKSBlowfish
    if len(password) > 72:
        raise ValueError('The password is too long. It must be 72 bytes at most.')
    if not 4 <= cost <= 31:
        raise ValueError('bcrypt cost factor must be in the range 4..31')
    cipher = _EKSBlowfish.new(password, _EKSBlowfish.MODE_ECB, salt, cost, invert)
    ctext = constant
    for _ in range(64):
        ctext = cipher.encrypt(ctext)
    return ctext