import xcffib
import struct
import io
from . import xproto
from . import shm
class EncodingInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.encoding, self.name_size, self.width, self.height = unpacker.unpack('IHHH2x')
        self.rate = Rational(unpacker)
        unpacker.pad('c')
        self.name = xcffib.List(unpacker, 'c', self.name_size)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=IHHH2x', self.encoding, self.name_size, self.width, self.height))
        buf.write(self.rate.pack() if hasattr(self.rate, 'pack') else Rational.synthetic(*self.rate).pack())
        buf.write(xcffib.pack_list(self.name, 'c'))
        buf.write(struct.pack('=4x'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, encoding, name_size, width, height, rate, name):
        self = cls.__new__(cls)
        self.encoding = encoding
        self.name_size = name_size
        self.width = width
        self.height = height
        self.rate = rate
        self.name = name
        return self