import xcffib
import struct
import io
class ListExtensionsReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.names_len, = unpacker.unpack('xB2x4x24x')
        self.names = xcffib.List(unpacker, STR, self.names_len)
        self.bufsize = unpacker.offset - base