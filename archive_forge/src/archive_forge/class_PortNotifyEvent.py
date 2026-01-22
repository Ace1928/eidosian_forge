import xcffib
import struct
import io
from . import xproto
from . import shm
class PortNotifyEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.time, self.port, self.attribute, self.value = unpacker.unpack('xx2xIIIi')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 1))
        buf.write(struct.pack('=x2xIIIi', self.time, self.port, self.attribute, self.value))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, time, port, attribute, value):
        self = cls.__new__(cls)
        self.time = time
        self.port = port
        self.attribute = attribute
        self.value = value
        return self