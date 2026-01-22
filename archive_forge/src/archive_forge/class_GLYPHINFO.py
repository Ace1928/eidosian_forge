import xcffib
import struct
import io
from . import xproto
class GLYPHINFO(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.width, self.height, self.x, self.y, self.x_off, self.y_off = unpacker.unpack('HHhhhh')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHhhhh', self.width, self.height, self.x, self.y, self.x_off, self.y_off))
        return buf.getvalue()
    fixed_size = 12

    @classmethod
    def synthetic(cls, width, height, x, y, x_off, y_off):
        self = cls.__new__(cls)
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.x_off = x_off
        self.y_off = y_off
        return self