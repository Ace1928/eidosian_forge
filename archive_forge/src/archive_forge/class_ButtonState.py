import xcffib
import struct
import io
from . import xfixes
from . import xproto
class ButtonState(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.class_id, self.len, self.num_buttons = unpacker.unpack('BBBx')
        self.buttons = xcffib.List(unpacker, 'B', 32)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BBBx', self.class_id, self.len, self.num_buttons))
        buf.write(xcffib.pack_list(self.buttons, 'B'))
        return buf.getvalue()
    fixed_size = 36

    @classmethod
    def synthetic(cls, class_id, len, num_buttons, buttons):
        self = cls.__new__(cls)
        self.class_id = class_id
        self.len = len
        self.num_buttons = num_buttons
        self.buttons = buttons
        return self