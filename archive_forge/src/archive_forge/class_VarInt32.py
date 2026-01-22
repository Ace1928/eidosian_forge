import struct
from struct import error
from .abstract import AbstractType
class VarInt32(AbstractType):

    @classmethod
    def decode(cls, data):
        value = UnsignedVarInt32.decode(data)
        return value >> 1 ^ -(value & 1)

    @classmethod
    def encode(cls, value):
        value &= 4294967295
        return UnsignedVarInt32.encode(value << 1 ^ value >> 31)