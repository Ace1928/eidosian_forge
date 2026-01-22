from Cryptodome.Util.py3compat import bchr, concat_buffers
from Cryptodome.Util._raw_api import (VoidPointer, SmartPointer,
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.Hash.keccak import _raw_keccak_lib
def _left_encode(x):
    """Left encode function as defined in NIST SP 800-185"""
    assert x < 1 << 2040 and x >= 0
    num = 1 if x == 0 else (x.bit_length() + 7) // 8
    return bchr(num) + long_to_bytes(x)