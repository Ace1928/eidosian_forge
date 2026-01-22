import xcffib
import struct
import io
from . import xproto
class QueryCounterReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        unpacker.unpack('xx2x4x')
        self.counter_value = INT64(unpacker)
        self.bufsize = unpacker.offset - base