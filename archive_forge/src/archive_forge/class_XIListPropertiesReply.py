import xcffib
import struct
import io
from . import xfixes
from . import xproto
class XIListPropertiesReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.num_properties, = unpacker.unpack('xx2x4xH22x')
        self.properties = xcffib.List(unpacker, 'I', self.num_properties)
        self.bufsize = unpacker.offset - base