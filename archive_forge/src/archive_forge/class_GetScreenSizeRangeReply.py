import xcffib
import struct
import io
from . import xproto
from . import render
class GetScreenSizeRangeReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.min_width, self.min_height, self.max_width, self.max_height = unpacker.unpack('xx2x4xHHHH16x')
        self.bufsize = unpacker.offset - base