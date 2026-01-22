import xcffib
import struct
import io
from . import xfixes
from . import xproto
class GroupInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.base, self.latched, self.locked, self.effective = unpacker.unpack('BBBB')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BBBB', self.base, self.latched, self.locked, self.effective))
        return buf.getvalue()
    fixed_size = 4

    @classmethod
    def synthetic(cls, base, latched, locked, effective):
        self = cls.__new__(cls)
        self.base = base
        self.latched = latched
        self.locked = locked
        self.effective = effective
        return self