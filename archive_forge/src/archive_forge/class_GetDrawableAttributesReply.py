import xcffib
import struct
import io
from . import xproto
class GetDrawableAttributesReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.num_attribs, = unpacker.unpack('xx2x4xI20x')
        self.attribs = xcffib.List(unpacker, 'I', self.num_attribs * 2)
        self.bufsize = unpacker.offset - base