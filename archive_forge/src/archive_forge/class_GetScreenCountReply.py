import xcffib
import struct
import io
from . import xproto
class GetScreenCountReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.screen_count, self.window = unpacker.unpack('xB2x4xI')
        self.bufsize = unpacker.offset - base