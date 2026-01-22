import xcffib
import struct
import io
from . import xfixes
from . import xproto
class BarrierHitEvent(xcffib.Event):
    xge = True

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.deviceid, self.time, self.eventid, self.root, self.event, self.barrier, self.dtime, self.flags, self.sourceid, self.root_x, self.root_y = unpacker.unpack('xx2xHIIIIIIIH2xii')
        self.dx = FP3232(unpacker)
        unpacker.pad(FP3232)
        self.dy = FP3232(unpacker)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 25))
        buf.write(struct.pack('=x2xHIIIIIIIH2xii', self.deviceid, self.time, self.eventid, self.root, self.event, self.barrier, self.dtime, self.flags, self.sourceid, self.root_x, self.root_y))
        buf.write(self.dx.pack() if hasattr(self.dx, 'pack') else FP3232.synthetic(*self.dx).pack())
        buf.write(self.dy.pack() if hasattr(self.dy, 'pack') else FP3232.synthetic(*self.dy).pack())
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, deviceid, time, eventid, root, event, barrier, dtime, flags, sourceid, root_x, root_y, dx, dy):
        self = cls.__new__(cls)
        self.deviceid = deviceid
        self.time = time
        self.eventid = eventid
        self.root = root
        self.event = event
        self.barrier = barrier
        self.dtime = dtime
        self.flags = flags
        self.sourceid = sourceid
        self.root_x = root_x
        self.root_y = root_y
        self.dx = dx
        self.dy = dy
        return self