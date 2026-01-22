import xcffib
import struct
import io
from . import xproto
from . import render
class RefreshRates(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.nRates, = unpacker.unpack('H')
        self.rates = xcffib.List(unpacker, 'H', self.nRates)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=H', self.nRates))
        buf.write(xcffib.pack_list(self.rates, 'H'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, nRates, rates):
        self = cls.__new__(cls)
        self.nRates = nRates
        self.rates = rates
        return self