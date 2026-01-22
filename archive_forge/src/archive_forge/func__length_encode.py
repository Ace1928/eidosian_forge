from Cryptodome.Util.number import long_to_bytes
from Cryptodome.Util.py3compat import bchr
from . import TurboSHAKE128
def _length_encode(x):
    if x == 0:
        return b'\x00'
    S = long_to_bytes(x)
    return S + bchr(len(S))