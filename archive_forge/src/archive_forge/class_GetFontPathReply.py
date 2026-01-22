import xcffib
import struct
import io
class GetFontPathReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.path_len, = unpacker.unpack('xx2x4xH22x')
        self.path = xcffib.List(unpacker, STR, self.path_len)
        self.bufsize = unpacker.offset - base