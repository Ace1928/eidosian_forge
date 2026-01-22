import xcffib
import struct
import io
from . import xproto
class QueryClientsReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.num_clients, = unpacker.unpack('xx2x4xI20x')
        self.clients = xcffib.List(unpacker, Client, self.num_clients)
        self.bufsize = unpacker.offset - base