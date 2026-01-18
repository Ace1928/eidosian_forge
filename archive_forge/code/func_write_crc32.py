from struct import pack
from binascii import crc32
def write_crc32(self, datum):
    data = crc32(datum) & 4294967295
    self._fo.write(pack('>I', data))