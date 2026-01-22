import xcffib
import struct
import io
from . import xproto
from . import render
class GetOutputPropertyReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.format, self.type, self.bytes_after, self.num_items = unpacker.unpack('xB2x4xIII12x')
        self.data = xcffib.List(unpacker, 'B', self.num_items * (self.format // 8))
        self.bufsize = unpacker.offset - base