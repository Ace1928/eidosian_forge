import xcffib
import struct
import io
from . import xfixes
from . import xproto
class InputInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.class_id, self.len = unpacker.unpack('BB')
        if self.class_id == InputClass.Key:
            self.min_keycode, self.max_keycode, self.num_keys = unpacker.unpack('BBH2x')
        if self.class_id == InputClass.Button:
            self.num_buttons, = unpacker.unpack('H')
        if self.class_id == InputClass.Valuator:
            self.axes_len, self.mode, self.motion_size = unpacker.unpack('BBI')
            self.axes = xcffib.List(unpacker, AxisInfo, self.axes_len)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BB', self.class_id, self.len))
        if self.class_id & InputClass.Key:
            self.min_keycode = self.info.pop(0)
            self.max_keycode = self.info.pop(0)
            self.num_keys = self.info.pop(0)
            buf.write(struct.pack('=BBH2x', self.min_keycode, self.max_keycode, self.num_keys))
        if self.class_id & InputClass.Button:
            self.num_buttons = self.info.pop(0)
            buf.write(struct.pack('=H', self.num_buttons))
        if self.class_id & InputClass.Valuator:
            self.axes_len = self.info.pop(0)
            self.mode = self.info.pop(0)
            self.motion_size = self.info.pop(0)
            self.axes = self.info.pop(0)
            buf.write(struct.pack('=BBI', self.axes_len, self.mode, self.motion_size))
            buf.write(xcffib.pack_list(self.axes, AxisInfo))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, class_id, len, info):
        self = cls.__new__(cls)
        self.class_id = class_id
        self.len = len
        self.info = info
        return self