import xcffib
import struct
import io
class AllocColorReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.red, self.green, self.blue, self.pixel = unpacker.unpack('xx2x4xHHH2xI')
        self.bufsize = unpacker.offset - base