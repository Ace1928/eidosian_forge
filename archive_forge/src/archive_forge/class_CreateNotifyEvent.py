import xcffib
import struct
import io
class CreateNotifyEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.parent, self.window, self.x, self.y, self.width, self.height, self.border_width, self.override_redirect = unpacker.unpack('xx2xIIhhHHHBx')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 16))
        buf.write(struct.pack('=x2xIIhhHHHBx', self.parent, self.window, self.x, self.y, self.width, self.height, self.border_width, self.override_redirect))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, parent, window, x, y, width, height, border_width, override_redirect):
        self = cls.__new__(cls)
        self.parent = parent
        self.window = window
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.border_width = border_width
        self.override_redirect = override_redirect
        return self