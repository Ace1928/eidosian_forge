import xcffib
import struct
import io
from . import xproto
class QueryScreensReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.number, = unpacker.unpack('xx2x4xI20x')
        self.screen_info = xcffib.List(unpacker, ScreenInfo, self.number)
        self.bufsize = unpacker.offset - base