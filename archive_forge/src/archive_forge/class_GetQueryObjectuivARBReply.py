import xcffib
import struct
import io
from . import xproto
class GetQueryObjectuivARBReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.n, self.datum = unpacker.unpack('xx2x4x4xII12x')
        self.data = xcffib.List(unpacker, 'I', self.n)
        self.bufsize = unpacker.offset - base