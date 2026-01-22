import xcffib
import struct
import io
from . import xfixes
from . import xproto
class XIGetSelectedEventsReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.num_masks, = unpacker.unpack('xx2x4xH22x')
        self.masks = xcffib.List(unpacker, EventMask, self.num_masks)
        self.bufsize = unpacker.offset - base