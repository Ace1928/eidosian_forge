import xcffib
import struct
import io
class QueryExtensionReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.present, self.major_opcode, self.first_event, self.first_error = unpacker.unpack('xx2x4xBBBB')
        self.bufsize = unpacker.offset - base