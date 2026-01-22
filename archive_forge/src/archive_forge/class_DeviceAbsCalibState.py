import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DeviceAbsCalibState(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.control_id, self.len, self.min_x, self.max_x, self.min_y, self.max_y, self.flip_x, self.flip_y, self.rotation, self.button_threshold = unpacker.unpack('HHiiiiIIII')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHiiiiIIII', self.control_id, self.len, self.min_x, self.max_x, self.min_y, self.max_y, self.flip_x, self.flip_y, self.rotation, self.button_threshold))
        return buf.getvalue()
    fixed_size = 36

    @classmethod
    def synthetic(cls, control_id, len, min_x, max_x, min_y, max_y, flip_x, flip_y, rotation, button_threshold):
        self = cls.__new__(cls)
        self.control_id = control_id
        self.len = len
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.rotation = rotation
        self.button_threshold = button_threshold
        return self