import xcffib
import struct
import io
from . import xproto
class GetVisualInfoReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.n_supported_visuals, = unpacker.unpack('xx2x4xI20x')
        self.supported_visuals = xcffib.List(unpacker, VisualInfos, self.n_supported_visuals)
        self.bufsize = unpacker.offset - base