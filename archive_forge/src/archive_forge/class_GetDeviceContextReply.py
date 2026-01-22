import xcffib
import struct
import io
from . import xproto
class GetDeviceContextReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.context_len, = unpacker.unpack('xx2x4xI20x')
        self.context = xcffib.List(unpacker, 'c', self.context_len)
        self.bufsize = unpacker.offset - base