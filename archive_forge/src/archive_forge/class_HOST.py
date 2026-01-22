import xcffib
import struct
import io
class HOST(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.family, self.address_len = unpacker.unpack('BxH')
        self.address = xcffib.List(unpacker, 'B', self.address_len)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BxH', self.family, self.address_len))
        buf.write(xcffib.pack_list(self.address, 'B'))
        buf.write(struct.pack('=4x'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, family, address_len, address):
        self = cls.__new__(cls)
        self.family = family
        self.address_len = address_len
        self.address = address
        return self