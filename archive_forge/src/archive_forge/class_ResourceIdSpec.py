import xcffib
import struct
import io
from . import xproto
class ResourceIdSpec(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.resource, self.type = unpacker.unpack('II')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=II', self.resource, self.type))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, resource, type):
        self = cls.__new__(cls)
        self.resource = resource
        self.type = type
        return self