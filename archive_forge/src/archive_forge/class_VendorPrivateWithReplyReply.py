import xcffib
import struct
import io
from . import xproto
class VendorPrivateWithReplyReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.retval, = unpacker.unpack('xx2x4xI')
        self.data1 = xcffib.List(unpacker, 'B', 24)
        unpacker.pad('B')
        self.data2 = xcffib.List(unpacker, 'B', self.length * 4)
        self.bufsize = unpacker.offset - base