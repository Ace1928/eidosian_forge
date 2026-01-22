import xcffib
import struct
import io
from . import xproto
class VisualInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.visual_id, self.depth, self.perf_level = unpacker.unpack('IBB2x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=IBB2x', self.visual_id, self.depth, self.perf_level))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, visual_id, depth, perf_level):
        self = cls.__new__(cls)
        self.visual_id = visual_id
        self.depth = depth
        self.perf_level = perf_level
        return self