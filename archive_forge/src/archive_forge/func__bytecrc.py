import struct
import sys
def _bytecrc(crc, poly, n):
    crc = long(crc)
    poly = long(poly)
    mask = long(1) << n - 1
    for i in xrange(8):
        if crc & mask:
            crc = crc << 1 ^ poly
        else:
            crc = crc << 1
    mask = (long(1) << n) - 1
    crc = crc & mask
    if mask <= sys.maxint:
        return int(crc)
    return crc