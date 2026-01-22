import xcffib
import struct
import io
from . import xfixes
from . import xproto
class KbdFeedbackState(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.class_id, self.feedback_id, self.len, self.pitch, self.duration, self.led_mask, self.led_values, self.global_auto_repeat, self.click, self.percent = unpacker.unpack('BBHHHIIBBBx')
        self.auto_repeats = xcffib.List(unpacker, 'B', 32)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BBHHHIIBBBx', self.class_id, self.feedback_id, self.len, self.pitch, self.duration, self.led_mask, self.led_values, self.global_auto_repeat, self.click, self.percent))
        buf.write(xcffib.pack_list(self.auto_repeats, 'B'))
        return buf.getvalue()
    fixed_size = 52

    @classmethod
    def synthetic(cls, class_id, feedback_id, len, pitch, duration, led_mask, led_values, global_auto_repeat, click, percent, auto_repeats):
        self = cls.__new__(cls)
        self.class_id = class_id
        self.feedback_id = feedback_id
        self.len = len
        self.pitch = pitch
        self.duration = duration
        self.led_mask = led_mask
        self.led_values = led_values
        self.global_auto_repeat = global_auto_repeat
        self.click = click
        self.percent = percent
        self.auto_repeats = auto_repeats
        return self