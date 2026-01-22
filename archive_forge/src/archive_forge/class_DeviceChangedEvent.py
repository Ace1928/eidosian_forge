import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DeviceChangedEvent(xcffib.Event):
    xge = True

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.deviceid, self.time, self.num_classes, self.sourceid, self.reason = unpacker.unpack('xx2xHIHHB11x')
        self.classes = xcffib.List(unpacker, DeviceClass, self.num_classes)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 1))
        buf.write(struct.pack('=x2xHIHHB11x', self.deviceid, self.time, self.num_classes, self.sourceid, self.reason))
        buf.write(xcffib.pack_list(self.classes, DeviceClass))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, deviceid, time, num_classes, sourceid, reason, classes):
        self = cls.__new__(cls)
        self.deviceid = deviceid
        self.time = time
        self.num_classes = num_classes
        self.sourceid = sourceid
        self.reason = reason
        self.classes = classes
        return self