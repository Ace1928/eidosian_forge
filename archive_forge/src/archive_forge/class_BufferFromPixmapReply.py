import xcffib
import struct
import io
from . import xproto
class BufferFromPixmapReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.nfd, self.size, self.width, self.height, self.stride, self.depth, self.bpp = unpacker.unpack('xB2x4xIHHHBB12x')
        self.bufsize = unpacker.offset - base