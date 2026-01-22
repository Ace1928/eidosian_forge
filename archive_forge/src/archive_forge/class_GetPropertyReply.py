import xcffib
import struct
import io
class GetPropertyReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.format, self.type, self.bytes_after, self.value_len = unpacker.unpack('xB2x4xIII12x')
        self.value = xcffib.List(unpacker, 'c', self.value_len * (self.format // 8))
        self.bufsize = unpacker.offset - base