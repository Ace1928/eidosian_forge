import xcffib
import struct
import io
class POINT(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.x, self.y = unpacker.unpack('hh')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=hh', self.x, self.y))
        return buf.getvalue()
    fixed_size = 4

    @classmethod
    def synthetic(cls, x, y):
        self = cls.__new__(cls)
        self.x = x
        self.y = y
        return self