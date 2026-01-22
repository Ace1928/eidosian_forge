import xcffib
import struct
import io
class GetModifierMappingReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.keycodes_per_modifier, = unpacker.unpack('xB2x4x24x')
        self.keycodes = xcffib.List(unpacker, 'B', self.keycodes_per_modifier * 8)
        self.bufsize = unpacker.offset - base