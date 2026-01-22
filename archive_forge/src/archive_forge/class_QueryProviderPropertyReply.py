import xcffib
import struct
import io
from . import xproto
from . import render
class QueryProviderPropertyReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.pending, self.range, self.immutable = unpacker.unpack('xx2x4xBBB21x')
        self.valid_values = xcffib.List(unpacker, 'i', self.length)
        self.bufsize = unpacker.offset - base