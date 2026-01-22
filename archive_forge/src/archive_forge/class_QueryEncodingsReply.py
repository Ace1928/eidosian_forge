import xcffib
import struct
import io
from . import xproto
from . import shm
class QueryEncodingsReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.num_encodings, = unpacker.unpack('xx2x4xH22x')
        self.info = xcffib.List(unpacker, EncodingInfo, self.num_encodings)
        self.bufsize = unpacker.offset - base