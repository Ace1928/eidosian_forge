import xcffib
import struct
import io
from . import xproto
from . import shm
class QueryAdaptorsReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.num_adaptors, = unpacker.unpack('xx2x4xH22x')
        self.info = xcffib.List(unpacker, AdaptorInfo, self.num_adaptors)
        self.bufsize = unpacker.offset - base