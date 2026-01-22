import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DeviceAbsAreaCtrl(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.control_id, self.len, self.offset_x, self.offset_y, self.width, self.height, self.screen, self.following = unpacker.unpack('HHIIiiiI')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHIIiiiI', self.control_id, self.len, self.offset_x, self.offset_y, self.width, self.height, self.screen, self.following))
        return buf.getvalue()
    fixed_size = 28

    @classmethod
    def synthetic(cls, control_id, len, offset_x, offset_y, width, height, screen, following):
        self = cls.__new__(cls)
        self.control_id = control_id
        self.len = len
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.width = width
        self.height = height
        self.screen = screen
        self.following = following
        return self