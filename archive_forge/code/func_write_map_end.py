from struct import pack
from binascii import crc32
def write_map_end(self):
    self.write_long(0)