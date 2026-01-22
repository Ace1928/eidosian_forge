import struct
from cryptography.utils import int_to_bytes
from twisted.python.deprecate import deprecated
from twisted.python.versions import Version
def MP(number):
    if number == 0:
        return b'\x00' * 4
    assert number > 0
    bn = int_to_bytes(number)
    if ord(bn[0:1]) & 128:
        bn = b'\x00' + bn
    return struct.pack('>L', len(bn)) + bn