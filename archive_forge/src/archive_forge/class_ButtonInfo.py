import xcffib
import struct
import io
from . import xfixes
from . import xproto
class ButtonInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.class_id, self.len, self.num_buttons = unpacker.unpack('BBH')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BBH', self.class_id, self.len, self.num_buttons))
        return buf.getvalue()
    fixed_size = 4

    @classmethod
    def synthetic(cls, class_id, len, num_buttons):
        self = cls.__new__(cls)
        self.class_id = class_id
        self.len = len
        self.num_buttons = num_buttons
        return self