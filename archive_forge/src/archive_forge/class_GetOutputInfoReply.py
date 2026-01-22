import xcffib
import struct
import io
from . import xproto
from . import render
class GetOutputInfoReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.status, self.timestamp, self.crtc, self.mm_width, self.mm_height, self.connection, self.subpixel_order, self.num_crtcs, self.num_modes, self.num_preferred, self.num_clones, self.name_len = unpacker.unpack('xB2x4xIIIIBBHHHHH')
        self.crtcs = xcffib.List(unpacker, 'I', self.num_crtcs)
        unpacker.pad('I')
        self.modes = xcffib.List(unpacker, 'I', self.num_modes)
        unpacker.pad('I')
        self.clones = xcffib.List(unpacker, 'I', self.num_clones)
        unpacker.pad('B')
        self.name = xcffib.List(unpacker, 'B', self.name_len)
        self.bufsize = unpacker.offset - base