import xcffib
import struct
import io
from . import xproto
class ListSelectionsReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.selections_len, = unpacker.unpack('xx2x4xI20x')
        self.selections = xcffib.List(unpacker, ListItem, self.selections_len)
        self.bufsize = unpacker.offset - base