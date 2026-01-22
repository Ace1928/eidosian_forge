import xcffib
import struct
import io
class GetClientDriverNameReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.client_driver_major_version, self.client_driver_minor_version, self.client_driver_patch_version, self.client_driver_name_len = unpacker.unpack('xx2x4xIIII8x')
        self.client_driver_name = xcffib.List(unpacker, 'c', self.client_driver_name_len)
        self.bufsize = unpacker.offset - base