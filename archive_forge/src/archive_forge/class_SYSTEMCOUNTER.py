import xcffib
import struct
import io
from . import xproto
class SYSTEMCOUNTER(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.counter, = unpacker.unpack('I')
        self.resolution = INT64(unpacker)
        self.name_len, = unpacker.unpack('H')
        unpacker.pad('c')
        self.name = xcffib.List(unpacker, 'c', self.name_len)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=I', self.counter))
        buf.write(self.resolution.pack() if hasattr(self.resolution, 'pack') else INT64.synthetic(*self.resolution).pack())
        buf.write(struct.pack('=H', self.name_len))
        buf.write(xcffib.pack_list(self.name, 'c'))
        buf.write(struct.pack('=4x'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, counter, resolution, name_len, name):
        self = cls.__new__(cls)
        self.counter = counter
        self.resolution = resolution
        self.name_len = name_len
        self.name = name
        return self