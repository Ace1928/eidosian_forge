from struct import pack
from binascii import crc32
def write_array_end(self):
    self.write_long(0)