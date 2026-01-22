import xcffib
import struct
import io
from . import xfixes
from . import xproto
class XIGrabDeviceReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.status, = unpacker.unpack('xx2x4xB23x')
        self.bufsize = unpacker.offset - base