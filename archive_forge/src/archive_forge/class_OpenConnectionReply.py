import xcffib
import struct
import io
class OpenConnectionReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.sarea_handle_low, self.sarea_handle_high, self.bus_id_len = unpacker.unpack('xx2x4xIII12x')
        self.bus_id = xcffib.List(unpacker, 'c', self.bus_id_len)
        self.bufsize = unpacker.offset - base