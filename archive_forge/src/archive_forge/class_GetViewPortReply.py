import xcffib
import struct
import io
class GetViewPortReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.x, self.y = unpacker.unpack('xx2x4xII16x')
        self.bufsize = unpacker.offset - base