import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DeviceStateNotifyEvent(xcffib.Event):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.device_id, self.time, self.num_keys, self.num_buttons, self.num_valuators, self.classes_reported = unpacker.unpack('xB2xIBBBB')
        self.buttons = xcffib.List(unpacker, 'B', 4)
        unpacker.pad('B')
        self.keys = xcffib.List(unpacker, 'B', 4)
        unpacker.pad('I')
        self.valuators = xcffib.List(unpacker, 'I', 3)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 10))
        buf.write(struct.pack('=B2xIBBBB', self.device_id, self.time, self.num_keys, self.num_buttons, self.num_valuators, self.classes_reported))
        buf.write(xcffib.pack_list(self.buttons, 'B'))
        buf.write(xcffib.pack_list(self.keys, 'B'))
        buf.write(xcffib.pack_list(self.valuators, 'I'))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, device_id, time, num_keys, num_buttons, num_valuators, classes_reported, buttons, keys, valuators):
        self = cls.__new__(cls)
        self.device_id = device_id
        self.time = time
        self.num_keys = num_keys
        self.num_buttons = num_buttons
        self.num_valuators = num_valuators
        self.classes_reported = classes_reported
        self.buttons = buttons
        self.keys = keys
        self.valuators = valuators
        return self