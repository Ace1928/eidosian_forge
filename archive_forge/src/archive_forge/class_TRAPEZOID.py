import xcffib
import struct
import io
from . import xproto
class TRAPEZOID(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.top, self.bottom = unpacker.unpack('ii')
        self.left = LINEFIX(unpacker)
        unpacker.pad(LINEFIX)
        self.right = LINEFIX(unpacker)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=ii', self.top, self.bottom))
        buf.write(self.left.pack() if hasattr(self.left, 'pack') else LINEFIX.synthetic(*self.left).pack())
        buf.write(self.right.pack() if hasattr(self.right, 'pack') else LINEFIX.synthetic(*self.right).pack())
        return buf.getvalue()

    @classmethod
    def synthetic(cls, top, bottom, left, right):
        self = cls.__new__(cls)
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        return self