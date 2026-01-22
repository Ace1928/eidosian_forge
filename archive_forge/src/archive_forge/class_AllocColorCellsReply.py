import xcffib
import struct
import io
class AllocColorCellsReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.pixels_len, self.masks_len = unpacker.unpack('xx2x4xHH20x')
        self.pixels = xcffib.List(unpacker, 'I', self.pixels_len)
        unpacker.pad('I')
        self.masks = xcffib.List(unpacker, 'I', self.masks_len)
        self.bufsize = unpacker.offset - base