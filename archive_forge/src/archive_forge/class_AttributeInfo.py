import xcffib
import struct
import io
from . import xproto
from . import shm
class AttributeInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.flags, self.min, self.max, self.size = unpacker.unpack('IiiI')
        self.name = xcffib.List(unpacker, 'c', self.size)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=IiiI', self.flags, self.min, self.max, self.size))
        buf.write(xcffib.pack_list(self.name, 'c'))
        buf.write(struct.pack('=4x'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, flags, min, max, size, name):
        self = cls.__new__(cls)
        self.flags = flags
        self.min = min
        self.max = max
        self.size = size
        self.name = name
        return self