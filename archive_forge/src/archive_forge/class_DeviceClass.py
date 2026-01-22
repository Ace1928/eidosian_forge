import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DeviceClass(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.type, self.len, self.sourceid = unpacker.unpack('HHH')
        if self.type == DeviceClassType.Key:
            self.num_keys, = unpacker.unpack('H')
            self.keys = xcffib.List(unpacker, 'I', self.num_keys)
        if self.type == DeviceClassType.Button:
            self.num_buttons, = unpacker.unpack('H')
            self.state = xcffib.List(unpacker, 'I', (self.num_buttons + 31) // 32)
            unpacker.pad('I')
            self.labels = xcffib.List(unpacker, 'I', self.num_buttons)
        if self.type == DeviceClassType.Valuator:
            self.number, self.label = unpacker.unpack('HI')
            self.min = FP3232(unpacker)
            unpacker.pad(FP3232)
            self.max = FP3232(unpacker)
            unpacker.pad(FP3232)
            self.value = FP3232(unpacker)
            self.resolution, self.mode = unpacker.unpack('IB3x')
        if self.type == DeviceClassType.Scroll:
            self.number, self.scroll_type, self.flags = unpacker.unpack('HH2xI')
            self.increment = FP3232(unpacker)
        if self.type == DeviceClassType.Touch:
            self.mode, self.num_touches = unpacker.unpack('BB')
        if self.type == DeviceClassType.Gesture:
            self.num_touches, = unpacker.unpack('Bx')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHH', self.type, self.len, self.sourceid))
        if self.type & DeviceClassType.Key:
            self.num_keys = self.data.pop(0)
            self.keys = self.data.pop(0)
            buf.write(struct.pack('=H', self.num_keys))
            buf.write(xcffib.pack_list(self.keys, 'I'))
        if self.type & DeviceClassType.Button:
            self.num_buttons = self.data.pop(0)
            self.state = self.data.pop(0)
            self.labels = self.data.pop(0)
            buf.write(struct.pack('=H', self.num_buttons))
            buf.write(xcffib.pack_list(self.state, 'I'))
            buf.write(xcffib.pack_list(self.labels, 'I'))
        if self.type & DeviceClassType.Valuator:
            self.number = self.data.pop(0)
            self.label = self.data.pop(0)
            self.min = self.data.pop(0)
            self.max = self.data.pop(0)
            self.value = self.data.pop(0)
            self.resolution = self.data.pop(0)
            self.mode = self.data.pop(0)
            self.data.pop(0)
            buf.write(struct.pack('=HI', self.number, self.label))
            buf.write(self.min.pack() if hasattr(self.min, 'pack') else FP3232.synthetic(*self.min).pack())
            buf.write(self.max.pack() if hasattr(self.max, 'pack') else FP3232.synthetic(*self.max).pack())
            buf.write(self.value.pack() if hasattr(self.value, 'pack') else FP3232.synthetic(*self.value).pack())
            buf.write(struct.pack('=I', self.resolution))
            buf.write(struct.pack('=B', self.mode))
            buf.write(struct.pack('=3x'))
        if self.type & DeviceClassType.Scroll:
            self.number = self.data.pop(0)
            self.scroll_type = self.data.pop(0)
            self.flags = self.data.pop(0)
            self.increment = self.data.pop(0)
            buf.write(struct.pack('=HH2xI', self.number, self.scroll_type, self.flags))
            buf.write(self.increment.pack() if hasattr(self.increment, 'pack') else FP3232.synthetic(*self.increment).pack())
        if self.type & DeviceClassType.Touch:
            self.mode = self.data.pop(0)
            self.num_touches = self.data.pop(0)
            buf.write(struct.pack('=BB', self.mode, self.num_touches))
        if self.type & DeviceClassType.Gesture:
            self.num_touches = self.data.pop(0)
            buf.write(struct.pack('=Bx', self.num_touches))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, type, len, sourceid, data):
        self = cls.__new__(cls)
        self.type = type
        self.len = len
        self.sourceid = sourceid
        self.data = data
        return self