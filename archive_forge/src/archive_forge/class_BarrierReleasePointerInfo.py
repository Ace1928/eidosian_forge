import xcffib
import struct
import io
from . import xfixes
from . import xproto
class BarrierReleasePointerInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.deviceid, self.barrier, self.eventid = unpacker.unpack('H2xII')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=H2xII', self.deviceid, self.barrier, self.eventid))
        return buf.getvalue()
    fixed_size = 12

    @classmethod
    def synthetic(cls, deviceid, barrier, eventid):
        self = cls.__new__(cls)
        self.deviceid = deviceid
        self.barrier = barrier
        self.eventid = eventid
        return self