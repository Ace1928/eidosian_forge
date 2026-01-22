import xcffib
import struct
import io
from . import xproto
class TRAP(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.top = SPANFIX(unpacker)
        unpacker.pad(SPANFIX)
        self.bot = SPANFIX(unpacker)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(self.top.pack() if hasattr(self.top, 'pack') else SPANFIX.synthetic(*self.top).pack())
        buf.write(self.bot.pack() if hasattr(self.bot, 'pack') else SPANFIX.synthetic(*self.bot).pack())
        return buf.getvalue()

    @classmethod
    def synthetic(cls, top, bot):
        self = cls.__new__(cls)
        self.top = top
        self.bot = bot
        return self