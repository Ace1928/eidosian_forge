import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DeviceResolutionState(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.control_id, self.len, self.num_valuators = unpacker.unpack('HHI')
        self.resolution_values = xcffib.List(unpacker, 'I', self.num_valuators)
        unpacker.pad('I')
        self.resolution_min = xcffib.List(unpacker, 'I', self.num_valuators)
        unpacker.pad('I')
        self.resolution_max = xcffib.List(unpacker, 'I', self.num_valuators)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHI', self.control_id, self.len, self.num_valuators))
        buf.write(xcffib.pack_list(self.resolution_values, 'I'))
        buf.write(xcffib.pack_list(self.resolution_min, 'I'))
        buf.write(xcffib.pack_list(self.resolution_max, 'I'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, control_id, len, num_valuators, resolution_values, resolution_min, resolution_max):
        self = cls.__new__(cls)
        self.control_id = control_id
        self.len = len
        self.num_valuators = num_valuators
        self.resolution_values = resolution_values
        self.resolution_min = resolution_min
        self.resolution_max = resolution_max
        return self