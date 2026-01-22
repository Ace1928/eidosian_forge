import xcffib
import struct
import io
from . import xv
class SurfaceInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.id, self.chroma_format, self.pad0, self.max_width, self.max_height, self.subpicture_max_width, self.subpicture_max_height, self.mc_type, self.flags = unpacker.unpack('IHHHHHHII')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=IHHHHHHII', self.id, self.chroma_format, self.pad0, self.max_width, self.max_height, self.subpicture_max_width, self.subpicture_max_height, self.mc_type, self.flags))
        return buf.getvalue()
    fixed_size = 24

    @classmethod
    def synthetic(cls, id, chroma_format, pad0, max_width, max_height, subpicture_max_width, subpicture_max_height, mc_type, flags):
        self = cls.__new__(cls)
        self.id = id
        self.chroma_format = chroma_format
        self.pad0 = pad0
        self.max_width = max_width
        self.max_height = max_height
        self.subpicture_max_width = subpicture_max_width
        self.subpicture_max_height = subpicture_max_height
        self.mc_type = mc_type
        self.flags = flags
        return self