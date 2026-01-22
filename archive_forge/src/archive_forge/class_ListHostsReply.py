import xcffib
import struct
import io
class ListHostsReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.mode, self.hosts_len = unpacker.unpack('xB2x4xH22x')
        self.hosts = xcffib.List(unpacker, HOST, self.hosts_len)
        self.bufsize = unpacker.offset - base