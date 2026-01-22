import xcffib
import struct
import io
from . import xproto
from . import render
class LeaseNotify(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.timestamp, self.window, self.lease, self.created = unpacker.unpack('IIIB15x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=IIIB15x', self.timestamp, self.window, self.lease, self.created))
        return buf.getvalue()
    fixed_size = 28

    @classmethod
    def synthetic(cls, timestamp, window, lease, created):
        self = cls.__new__(cls)
        self.timestamp = timestamp
        self.window = window
        self.lease = lease
        self.created = created
        return self