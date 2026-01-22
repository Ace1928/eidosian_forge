import xcffib
import struct
import io
from . import xproto
from . import render
class GetPanningReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.status, self.timestamp, self.left, self.top, self.width, self.height, self.track_left, self.track_top, self.track_width, self.track_height, self.border_left, self.border_top, self.border_right, self.border_bottom = unpacker.unpack('xB2x4xIHHHHHHHHhhhh')
        self.bufsize = unpacker.offset - base