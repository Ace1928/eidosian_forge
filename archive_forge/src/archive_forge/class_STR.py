import xcffib
import struct
import io
class STR(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.name_len, = unpacker.unpack('B')
        self.name = xcffib.List(unpacker, 'c', self.name_len)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', self.name_len))
        buf.write(xcffib.pack_list(self.name, 'c'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, name_len, name):
        self = cls.__new__(cls)
        self.name_len = name_len
        self.name = name
        return self