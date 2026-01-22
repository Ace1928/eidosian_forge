import xcffib
import struct
import io
class GetAllModeLinesReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.modecount, = unpacker.unpack('xx2x4xI20x')
        self.modeinfo = xcffib.List(unpacker, ModeInfo, self.modecount)
        self.bufsize = unpacker.offset - base