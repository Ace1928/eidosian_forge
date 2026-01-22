import xcffib
import struct
import io
class SetupAuthenticate(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.status, self.length = unpacker.unpack('B5xH')
        self.reason = xcffib.List(unpacker, 'c', self.length * 4)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B5xH', self.status, self.length))
        buf.write(xcffib.pack_list(self.reason, 'c'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, status, length, reason):
        self = cls.__new__(cls)
        self.status = status
        self.length = length
        self.reason = reason
        return self