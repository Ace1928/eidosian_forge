import xcffib
import struct
import io
class ConfigureRequestEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.stack_mode, self.parent, self.window, self.sibling, self.x, self.y, self.width, self.height, self.border_width, self.value_mask = unpacker.unpack('xB2xIIIhhHHHH')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 23))
        buf.write(struct.pack('=B2xIIIhhHHHH', self.stack_mode, self.parent, self.window, self.sibling, self.x, self.y, self.width, self.height, self.border_width, self.value_mask))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, stack_mode, parent, window, sibling, x, y, width, height, border_width, value_mask):
        self = cls.__new__(cls)
        self.stack_mode = stack_mode
        self.parent = parent
        self.window = window
        self.sibling = sibling
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.border_width = border_width
        self.value_mask = value_mask
        return self