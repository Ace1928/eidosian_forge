import xcffib
import struct
import io
from . import xproto
class GetTimeoutsReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.standby_timeout, self.suspend_timeout, self.off_timeout = unpacker.unpack('xx2x4xHHH18x')
        self.bufsize = unpacker.offset - base