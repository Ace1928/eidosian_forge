import xcffib
import struct
import io
from . import xfixes
from . import xproto
class GetDeviceDontPropagateListReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.xi_reply_type, self.num_classes = unpacker.unpack('xB2x4xH22x')
        self.classes = xcffib.List(unpacker, 'I', self.num_classes)
        self.bufsize = unpacker.offset - base