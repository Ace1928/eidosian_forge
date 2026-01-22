import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DeviceCoreState(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.control_id, self.len, self.status, self.iscore = unpacker.unpack('HHBB2x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHBB2x', self.control_id, self.len, self.status, self.iscore))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, control_id, len, status, iscore):
        self = cls.__new__(cls)
        self.control_id = control_id
        self.len = len
        self.status = status
        self.iscore = iscore
        return self