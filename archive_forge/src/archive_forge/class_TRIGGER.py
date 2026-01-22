import xcffib
import struct
import io
from . import xproto
class TRIGGER(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.counter, self.wait_type = unpacker.unpack('II')
        self.wait_value = INT64(unpacker)
        self.test_type, = unpacker.unpack('I')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=II', self.counter, self.wait_type))
        buf.write(self.wait_value.pack() if hasattr(self.wait_value, 'pack') else INT64.synthetic(*self.wait_value).pack())
        buf.write(struct.pack('=I', self.test_type))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, counter, wait_type, wait_value, test_type):
        self = cls.__new__(cls)
        self.counter = counter
        self.wait_type = wait_type
        self.wait_value = wait_value
        self.test_type = test_type
        return self