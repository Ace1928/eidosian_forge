import xcffib
import struct
import io
from . import xfixes
from . import xproto
class ValuatorInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.class_id, self.len, self.axes_len, self.mode, self.motion_size = unpacker.unpack('BBBBI')
        self.axes = xcffib.List(unpacker, AxisInfo, self.axes_len)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BBBBI', self.class_id, self.len, self.axes_len, self.mode, self.motion_size))
        buf.write(xcffib.pack_list(self.axes, AxisInfo))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, class_id, len, axes_len, mode, motion_size, axes):
        self = cls.__new__(cls)
        self.class_id = class_id
        self.len = len
        self.axes_len = axes_len
        self.mode = mode
        self.motion_size = motion_size
        self.axes = axes
        return self