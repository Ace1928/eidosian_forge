import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DevicePropertyNotifyEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.state, self.time, self.property, self.device_id = unpacker.unpack('xB2xII19xB')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 16))
        buf.write(struct.pack('=B2xII19xB', self.state, self.time, self.property, self.device_id))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, state, time, property, device_id):
        self = cls.__new__(cls)
        self.state = state
        self.time = time
        self.property = property
        self.device_id = device_id
        return self