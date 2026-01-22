import xcffib
import struct
import io
class GetScreenSaverReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.timeout, self.interval, self.prefer_blanking, self.allow_exposures = unpacker.unpack('xx2x4xHHBB18x')
        self.bufsize = unpacker.offset - base