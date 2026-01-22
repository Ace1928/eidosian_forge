import xcffib
import struct
import io
from . import xproto
class QueryClientResourcesReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.num_types, = unpacker.unpack('xx2x4xI20x')
        self.types = xcffib.List(unpacker, Type, self.num_types)
        self.bufsize = unpacker.offset - base