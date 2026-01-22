import xcffib
import struct
import io
from . import xproto
class AttachFormat(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.attachment, self.format = unpacker.unpack('II')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=II', self.attachment, self.format))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, attachment, format):
        self = cls.__new__(cls)
        self.attachment = attachment
        self.format = format
        return self