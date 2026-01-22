import xcffib
import struct
import io
from . import xproto
class POINTFIX(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.x, self.y = unpacker.unpack('ii')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=ii', self.x, self.y))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, x, y):
        self = cls.__new__(cls)
        self.x = x
        self.y = y
        return self