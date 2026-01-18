from struct import pack
from binascii import crc32
def write_item_count(self, length):
    self.write_long(length)