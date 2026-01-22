import xcffib
import struct
import io
from . import xfixes
from . import xproto
class XIQueryPointerReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.root, self.child, self.root_x, self.root_y, self.win_x, self.win_y, self.same_screen, self.buttons_len = unpacker.unpack('xx2x4xIIiiiiBxH')
        self.mods = ModifierInfo(unpacker)
        unpacker.pad(GroupInfo)
        self.group = GroupInfo(unpacker)
        unpacker.pad('I')
        self.buttons = xcffib.List(unpacker, 'I', self.buttons_len)
        self.bufsize = unpacker.offset - base