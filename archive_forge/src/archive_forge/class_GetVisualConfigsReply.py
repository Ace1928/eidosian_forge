import xcffib
import struct
import io
from . import xproto
class GetVisualConfigsReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.num_visuals, self.num_properties = unpacker.unpack('xx2x4xII16x')
        self.property_list = xcffib.List(unpacker, 'I', self.length)
        self.bufsize = unpacker.offset - base