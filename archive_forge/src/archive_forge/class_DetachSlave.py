import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DetachSlave(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.type, self.len, self.deviceid = unpacker.unpack('HHH2x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHH2x', self.type, self.len, self.deviceid))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, type, len, deviceid):
        self = cls.__new__(cls)
        self.type = type
        self.len = len
        self.deviceid = deviceid
        return self