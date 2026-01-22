import xcffib
import struct
import io
from . import xfixes
from . import xproto
class XIQueryDeviceReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.num_infos, = unpacker.unpack('xx2x4xH22x')
        self.infos = xcffib.List(unpacker, XIDeviceInfo, self.num_infos)
        self.bufsize = unpacker.offset - base