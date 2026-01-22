import xcffib
import struct
import io
from . import xproto
class IsDirectReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.is_direct, = unpacker.unpack('xx2x4xB23x')
        self.bufsize = unpacker.offset - base