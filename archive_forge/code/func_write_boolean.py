from struct import pack
from binascii import crc32
def write_boolean(self, datum):
    self._fo.write(pack('B', 1 if datum else 0))