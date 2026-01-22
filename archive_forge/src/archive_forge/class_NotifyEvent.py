import xcffib
import struct
import io
from . import xproto
class NotifyEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.state, self.time, self.root, self.window, self.kind, self.forced = unpacker.unpack('xB2xIIIBB14x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 0))
        buf.write(struct.pack('=B2xIIIBB14x', self.state, self.time, self.root, self.window, self.kind, self.forced))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, state, time, root, window, kind, forced):
        self = cls.__new__(cls)
        self.state = state
        self.time = time
        self.root = root
        self.window = window
        self.kind = kind
        self.forced = forced
        return self