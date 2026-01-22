import xcffib
import struct
import io
from . import xfixes
from . import xproto
class RemoveMaster(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.type, self.len, self.deviceid, self.return_mode, self.return_pointer, self.return_keyboard = unpacker.unpack('HHHBxHH')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHHBxHH', self.type, self.len, self.deviceid, self.return_mode, self.return_pointer, self.return_keyboard))
        return buf.getvalue()
    fixed_size = 12

    @classmethod
    def synthetic(cls, type, len, deviceid, return_mode, return_pointer, return_keyboard):
        self = cls.__new__(cls)
        self.type = type
        self.len = len
        self.deviceid = deviceid
        self.return_mode = return_mode
        self.return_pointer = return_pointer
        self.return_keyboard = return_keyboard
        return self