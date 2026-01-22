import xcffib
import struct
import io
from . import xproto
class DRI2Buffer(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.attachment, self.name, self.pitch, self.cpp, self.flags = unpacker.unpack('IIIII')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=IIIII', self.attachment, self.name, self.pitch, self.cpp, self.flags))
        return buf.getvalue()
    fixed_size = 20

    @classmethod
    def synthetic(cls, attachment, name, pitch, cpp, flags):
        self = cls.__new__(cls)
        self.attachment = attachment
        self.name = name
        self.pitch = pitch
        self.cpp = cpp
        self.flags = flags
        return self