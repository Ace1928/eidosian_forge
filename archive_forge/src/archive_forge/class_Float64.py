import struct
from struct import error
from .abstract import AbstractType
class Float64(AbstractType):
    _pack = struct.Struct('>d').pack
    _unpack = struct.Struct('>d').unpack

    @classmethod
    def encode(cls, value):
        return _pack(cls._pack, value)

    @classmethod
    def decode(cls, data):
        return _unpack(cls._unpack, data.read(8))