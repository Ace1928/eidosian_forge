import xcffib
import struct
import io
from . import xproto
class PICTVISUAL(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.visual, self.format = unpacker.unpack('II')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=II', self.visual, self.format))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, visual, format):
        self = cls.__new__(cls)
        self.visual = visual
        self.format = format
        return self