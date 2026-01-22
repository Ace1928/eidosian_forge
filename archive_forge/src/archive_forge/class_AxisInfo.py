import xcffib
import struct
import io
from . import xfixes
from . import xproto
class AxisInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.resolution, self.minimum, self.maximum = unpacker.unpack('Iii')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=Iii', self.resolution, self.minimum, self.maximum))
        return buf.getvalue()
    fixed_size = 12

    @classmethod
    def synthetic(cls, resolution, minimum, maximum):
        self = cls.__new__(cls)
        self.resolution = resolution
        self.minimum = minimum
        self.maximum = maximum
        return self