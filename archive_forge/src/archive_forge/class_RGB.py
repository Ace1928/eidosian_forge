import xcffib
import struct
import io
class RGB(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.red, self.green, self.blue = unpacker.unpack('HHH2x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHH2x', self.red, self.green, self.blue))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, red, green, blue):
        self = cls.__new__(cls)
        self.red = red
        self.green = green
        self.blue = blue
        return self