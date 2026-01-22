import xcffib
import struct
import io
from . import xfixes
from . import xproto
class ButtonClass(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.type, self.len, self.sourceid, self.num_buttons = unpacker.unpack('HHHH')
        self.state = xcffib.List(unpacker, 'I', (self.num_buttons + 31) // 32)
        unpacker.pad('I')
        self.labels = xcffib.List(unpacker, 'I', self.num_buttons)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHHH', self.type, self.len, self.sourceid, self.num_buttons))
        buf.write(xcffib.pack_list(self.state, 'I'))
        buf.write(xcffib.pack_list(self.labels, 'I'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, type, len, sourceid, num_buttons, state, labels):
        self = cls.__new__(cls)
        self.type = type
        self.len = len
        self.sourceid = sourceid
        self.num_buttons = num_buttons
        self.state = state
        self.labels = labels
        return self