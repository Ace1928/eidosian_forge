import xcffib
import struct
import io
from . import xfixes
from . import xproto
class PropertyEvent(xcffib.Event):
    xge = True

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.deviceid, self.time, self.property, self.what = unpacker.unpack('xx2xHIIB11x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 12))
        buf.write(struct.pack('=x2xHIIB11x', self.deviceid, self.time, self.property, self.what))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, deviceid, time, property, what):
        self = cls.__new__(cls)
        self.deviceid = deviceid
        self.time = time
        self.property = property
        self.what = what
        return self