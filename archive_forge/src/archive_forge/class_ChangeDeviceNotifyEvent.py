import xcffib
import struct
import io
from . import xfixes
from . import xproto
class ChangeDeviceNotifyEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.device_id, self.time, self.request = unpacker.unpack('xB2xIB23x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 12))
        buf.write(struct.pack('=B2xIB23x', self.device_id, self.time, self.request))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, device_id, time, request):
        self = cls.__new__(cls)
        self.device_id = device_id
        self.time = time
        self.request = request
        return self