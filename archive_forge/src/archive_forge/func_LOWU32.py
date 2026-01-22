import struct
import zlib
def LOWU32(i):
    """Return the low-order 32 bits of an int, as a non-negative int."""
    return i & 4294967295