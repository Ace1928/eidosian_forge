import xcffib
import struct
import io
from . import xproto
class QueryInfoReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.state, self.saver_window, self.ms_until_server, self.ms_since_user_input, self.event_mask, self.kind = unpacker.unpack('xB2x4xIIIIB7x')
        self.bufsize = unpacker.offset - base