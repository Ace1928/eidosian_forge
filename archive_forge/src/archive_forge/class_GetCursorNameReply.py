import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
class GetCursorNameReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.atom, self.nbytes = unpacker.unpack('xx2x4xIH18x')
        self.name = xcffib.List(unpacker, 'c', self.nbytes)
        self.bufsize = unpacker.offset - base