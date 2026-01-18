from struct import pack
from binascii import crc32
def write_utf8(self, datum):
    try:
        encoded = datum.encode()
    except AttributeError:
        raise TypeError('must be string')
    self.write_bytes(encoded)