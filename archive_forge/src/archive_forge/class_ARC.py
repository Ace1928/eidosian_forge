import xcffib
import struct
import io
class ARC(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.x, self.y, self.width, self.height, self.angle1, self.angle2 = unpacker.unpack('hhHHhh')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=hhHHhh', self.x, self.y, self.width, self.height, self.angle1, self.angle2))
        return buf.getvalue()
    fixed_size = 12

    @classmethod
    def synthetic(cls, x, y, width, height, angle1, angle2):
        self = cls.__new__(cls)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.angle1 = angle1
        self.angle2 = angle2
        return self