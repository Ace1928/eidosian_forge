import xcffib
import struct
import io
from . import xproto
class GetPixelMapusvReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.n, self.datum = unpacker.unpack('xx2x4x4xIH16x')
        self.data = xcffib.List(unpacker, 'H', self.n)
        self.bufsize = unpacker.offset - base