import xcffib
import struct
import io
from . import xproto
class GetMapivReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.n, self.datum = unpacker.unpack('xx2x4x4xIi12x')
        self.data = xcffib.List(unpacker, 'i', self.n)
        self.bufsize = unpacker.offset - base