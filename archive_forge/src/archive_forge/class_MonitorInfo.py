import xcffib
import struct
import io
from . import xproto
from . import render
class MonitorInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.name, self.primary, self.automatic, self.nOutput, self.x, self.y, self.width, self.height, self.width_in_millimeters, self.height_in_millimeters = unpacker.unpack('IBBHhhHHII')
        self.outputs = xcffib.List(unpacker, 'I', self.nOutput)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=IBBHhhHHII', self.name, self.primary, self.automatic, self.nOutput, self.x, self.y, self.width, self.height, self.width_in_millimeters, self.height_in_millimeters))
        buf.write(xcffib.pack_list(self.outputs, 'I'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, name, primary, automatic, nOutput, x, y, width, height, width_in_millimeters, height_in_millimeters, outputs):
        self = cls.__new__(cls)
        self.name = name
        self.primary = primary
        self.automatic = automatic
        self.nOutput = nOutput
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.width_in_millimeters = width_in_millimeters
        self.height_in_millimeters = height_in_millimeters
        self.outputs = outputs
        return self