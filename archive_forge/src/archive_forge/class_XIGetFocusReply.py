import xcffib
import struct
import io
from . import xfixes
from . import xproto
class XIGetFocusReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.focus, = unpacker.unpack('xx2x4xI20x')
        self.bufsize = unpacker.offset - base