import xcffib
import struct
import io
from . import xproto
class PbufferClobberEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.event_type, self.draw_type, self.drawable, self.b_mask, self.aux_buffer, self.x, self.y, self.width, self.height, self.count = unpacker.unpack('xx2xHHIIHHHHHH4x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 0))
        buf.write(struct.pack('=x2xHHIIHHHHHH4x', self.event_type, self.draw_type, self.drawable, self.b_mask, self.aux_buffer, self.x, self.y, self.width, self.height, self.count))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, event_type, draw_type, drawable, b_mask, aux_buffer, x, y, width, height, count):
        self = cls.__new__(cls)
        self.event_type = event_type
        self.draw_type = draw_type
        self.drawable = drawable
        self.b_mask = b_mask
        self.aux_buffer = aux_buffer
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.count = count
        return self