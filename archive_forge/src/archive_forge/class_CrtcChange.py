import xcffib
import struct
import io
from . import xproto
from . import render
class CrtcChange(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.timestamp, self.window, self.crtc, self.mode, self.rotation, self.x, self.y, self.width, self.height = unpacker.unpack('IIIIH2xhhHH')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=IIIIH2xhhHH', self.timestamp, self.window, self.crtc, self.mode, self.rotation, self.x, self.y, self.width, self.height))
        return buf.getvalue()
    fixed_size = 28

    @classmethod
    def synthetic(cls, timestamp, window, crtc, mode, rotation, x, y, width, height):
        self = cls.__new__(cls)
        self.timestamp = timestamp
        self.window = window
        self.crtc = crtc
        self.mode = mode
        self.rotation = rotation
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        return self