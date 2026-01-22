from binascii import unhexlify
from Cryptodome.Hash import BLAKE2s
from Cryptodome.Util.strxor import strxor
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Util.py3compat import bord, tobytes, _copy_bytes
from Cryptodome.Random import get_random_bytes
def _shift_bytes(bs, xor_lsb=0):
    num = bytes_to_long(bs) << 1 ^ xor_lsb
    return long_to_bytes(num, len(bs))[-len(bs):]