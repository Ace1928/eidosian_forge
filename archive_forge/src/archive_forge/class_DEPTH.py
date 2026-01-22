import xcffib
import struct
import io
class DEPTH(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.depth, self.visuals_len = unpacker.unpack('BxH4x')
        self.visuals = xcffib.List(unpacker, VISUALTYPE, self.visuals_len)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BxH4x', self.depth, self.visuals_len))
        buf.write(xcffib.pack_list(self.visuals, VISUALTYPE))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, depth, visuals_len, visuals):
        self = cls.__new__(cls)
        self.depth = depth
        self.visuals_len = visuals_len
        self.visuals = visuals
        return self