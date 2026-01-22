import xcffib
import struct
import io
from . import xproto
class WAITCONDITION(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.trigger = TRIGGER(unpacker)
        unpacker.pad(INT64)
        self.event_threshold = INT64(unpacker)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(self.trigger.pack() if hasattr(self.trigger, 'pack') else TRIGGER.synthetic(*self.trigger).pack())
        buf.write(self.event_threshold.pack() if hasattr(self.event_threshold, 'pack') else INT64.synthetic(*self.event_threshold).pack())
        return buf.getvalue()

    @classmethod
    def synthetic(cls, trigger, event_threshold):
        self = cls.__new__(cls)
        self.trigger = trigger
        self.event_threshold = event_threshold
        return self