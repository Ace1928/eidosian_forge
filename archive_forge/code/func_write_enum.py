from struct import pack
from binascii import crc32
def write_enum(self, index):
    self.write_int(index)