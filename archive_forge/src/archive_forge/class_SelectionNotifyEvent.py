import xcffib
import struct
import io
class SelectionNotifyEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.time, self.requestor, self.selection, self.target, self.property = unpacker.unpack('xx2xIIIII')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 31))
        buf.write(struct.pack('=x2xIIIII', self.time, self.requestor, self.selection, self.target, self.property))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, time, requestor, selection, target, property):
        self = cls.__new__(cls)
        self.time = time
        self.requestor = requestor
        self.selection = selection
        self.target = target
        self.property = property
        return self