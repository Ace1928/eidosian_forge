import xcffib
import struct
import io
class AuthConnectionReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.authenticated, = unpacker.unpack('xx2x4xI')
        self.bufsize = unpacker.offset - base