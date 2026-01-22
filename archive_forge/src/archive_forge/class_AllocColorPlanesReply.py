import xcffib
import struct
import io
class AllocColorPlanesReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.pixels_len, self.red_mask, self.green_mask, self.blue_mask = unpacker.unpack('xx2x4xH2xIII8x')
        self.pixels = xcffib.List(unpacker, 'I', self.pixels_len)
        self.bufsize = unpacker.offset - base