import xcffib
import struct
import io
class QueryBestSizeReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.width, self.height = unpacker.unpack('xx2x4xHH')
        self.bufsize = unpacker.offset - base