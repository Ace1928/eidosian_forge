from struct import pack
from binascii import crc32
def write_fixed(self, datum):
    self._fo.write(datum)