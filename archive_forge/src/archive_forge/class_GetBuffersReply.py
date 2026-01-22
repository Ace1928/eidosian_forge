import xcffib
import struct
import io
from . import xproto
class GetBuffersReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.width, self.height, self.count = unpacker.unpack('xx2x4xIII12x')
        self.buffers = xcffib.List(unpacker, DRI2Buffer, self.count)
        self.bufsize = unpacker.offset - base