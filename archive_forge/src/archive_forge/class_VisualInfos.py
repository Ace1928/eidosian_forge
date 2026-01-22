import xcffib
import struct
import io
from . import xproto
class VisualInfos(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.n_infos, = unpacker.unpack('I')
        self.infos = xcffib.List(unpacker, VisualInfo, self.n_infos)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=I', self.n_infos))
        buf.write(xcffib.pack_list(self.infos, VisualInfo))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, n_infos, infos):
        self = cls.__new__(cls)
        self.n_infos = n_infos
        self.infos = infos
        return self