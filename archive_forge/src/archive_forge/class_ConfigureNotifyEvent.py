import xcffib
import struct
import io
class ConfigureNotifyEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.event, self.window, self.above_sibling, self.x, self.y, self.width, self.height, self.border_width, self.override_redirect = unpacker.unpack('xx2xIIIhhHHHBx')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 22))
        buf.write(struct.pack('=x2xIIIhhHHHBx', self.event, self.window, self.above_sibling, self.x, self.y, self.width, self.height, self.border_width, self.override_redirect))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, event, window, above_sibling, x, y, width, height, border_width, override_redirect):
        self = cls.__new__(cls)
        self.event = event
        self.window = window
        self.above_sibling = above_sibling
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.border_width = border_width
        self.override_redirect = override_redirect
        return self