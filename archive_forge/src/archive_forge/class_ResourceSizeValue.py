import xcffib
import struct
import io
from . import xproto
class ResourceSizeValue(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.size = ResourceSizeSpec(unpacker)
        self.num_cross_references, = unpacker.unpack('I')
        unpacker.pad(ResourceSizeSpec)
        self.cross_references = xcffib.List(unpacker, ResourceSizeSpec, self.num_cross_references)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(self.size.pack() if hasattr(self.size, 'pack') else ResourceSizeSpec.synthetic(*self.size).pack())
        buf.write(struct.pack('=I', self.num_cross_references))
        buf.write(xcffib.pack_list(self.cross_references, ResourceSizeSpec))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, size, num_cross_references, cross_references):
        self = cls.__new__(cls)
        self.size = size
        self.num_cross_references = num_cross_references
        self.cross_references = cross_references
        return self