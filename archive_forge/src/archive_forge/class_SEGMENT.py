import xcffib
import struct
import io
class SEGMENT(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.x1, self.y1, self.x2, self.y2 = unpacker.unpack('hhhh')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=hhhh', self.x1, self.y1, self.x2, self.y2))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, x1, y1, x2, y2):
        self = cls.__new__(cls)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        return self