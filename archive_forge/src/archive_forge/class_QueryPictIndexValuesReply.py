import xcffib
import struct
import io
from . import xproto
class QueryPictIndexValuesReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.num_values, = unpacker.unpack('xx2x4xI20x')
        self.values = xcffib.List(unpacker, INDEXVALUE, self.num_values)
        self.bufsize = unpacker.offset - base