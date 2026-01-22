import struct
from struct import error
from .abstract import AbstractType
class CompactString(String):

    def decode(self, data):
        length = UnsignedVarInt32.decode(data) - 1
        if length < 0:
            return None
        value = data.read(length)
        if len(value) != length:
            raise ValueError('Buffer underrun decoding string')
        return value.decode(self.encoding)

    def encode(self, value):
        if value is None:
            return UnsignedVarInt32.encode(0)
        value = str(value).encode(self.encoding)
        return UnsignedVarInt32.encode(len(value) + 1) + value