import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DevicePresenceNotifyEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.time, self.devchange, self.device_id, self.control = unpacker.unpack('xx2xIBBH20x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 15))
        buf.write(struct.pack('=x2xIBBH20x', self.time, self.devchange, self.device_id, self.control))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, time, devchange, device_id, control):
        self = cls.__new__(cls)
        self.time = time
        self.devchange = devchange
        self.device_id = device_id
        self.control = control
        return self