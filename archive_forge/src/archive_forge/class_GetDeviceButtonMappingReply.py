import xcffib
import struct
import io
from . import xfixes
from . import xproto
class GetDeviceButtonMappingReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.xi_reply_type, self.map_size = unpacker.unpack('xB2x4xB23x')
        self.map = xcffib.List(unpacker, 'B', self.map_size)
        self.bufsize = unpacker.offset - base