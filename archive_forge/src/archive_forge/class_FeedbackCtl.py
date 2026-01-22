import xcffib
import struct
import io
from . import xfixes
from . import xproto
class FeedbackCtl(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.class_id, self.feedback_id, self.len = unpacker.unpack('BBH')
        if self.class_id == FeedbackClass.Keyboard:
            self.key, self.auto_repeat_mode, self.key_click_percent, self.bell_percent, self.bell_pitch, self.bell_duration, self.led_mask, self.led_values = unpacker.unpack('BBbbhhII')
        if self.class_id == FeedbackClass.Pointer:
            self.num, self.denom, self.threshold = unpacker.unpack('2xhhh')
        if self.class_id == FeedbackClass.String:
            self.num_keysyms, = unpacker.unpack('2xH')
            self.keysyms = xcffib.List(unpacker, 'I', self.num_keysyms)
        if self.class_id == FeedbackClass.Integer:
            self.int_to_display, = unpacker.unpack('i')
        if self.class_id == FeedbackClass.Led:
            self.led_mask, self.led_values = unpacker.unpack('II')
        if self.class_id == FeedbackClass.Bell:
            self.percent, self.pitch, self.duration = unpacker.unpack('b3xhh')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BBH', self.class_id, self.feedback_id, self.len))
        if self.class_id & FeedbackClass.Keyboard:
            self.key = self.data.pop(0)
            self.auto_repeat_mode = self.data.pop(0)
            self.key_click_percent = self.data.pop(0)
            self.bell_percent = self.data.pop(0)
            self.bell_pitch = self.data.pop(0)
            self.bell_duration = self.data.pop(0)
            self.led_mask = self.data.pop(0)
            self.led_values = self.data.pop(0)
            buf.write(struct.pack('=BBbbhhII', self.key, self.auto_repeat_mode, self.key_click_percent, self.bell_percent, self.bell_pitch, self.bell_duration, self.led_mask, self.led_values))
        if self.class_id & FeedbackClass.Pointer:
            self.num = self.data.pop(0)
            self.denom = self.data.pop(0)
            self.threshold = self.data.pop(0)
            buf.write(struct.pack('=2xhhh', self.num, self.denom, self.threshold))
        if self.class_id & FeedbackClass.String:
            self.num_keysyms = self.data.pop(0)
            self.keysyms = self.data.pop(0)
            buf.write(struct.pack('=2xH', self.num_keysyms))
            buf.write(xcffib.pack_list(self.keysyms, 'I'))
        if self.class_id & FeedbackClass.Integer:
            self.int_to_display = self.data.pop(0)
            buf.write(struct.pack('=i', self.int_to_display))
        if self.class_id & FeedbackClass.Led:
            self.led_mask = self.data.pop(0)
            self.led_values = self.data.pop(0)
            buf.write(struct.pack('=II', self.led_mask, self.led_values))
        if self.class_id & FeedbackClass.Bell:
            self.percent = self.data.pop(0)
            self.pitch = self.data.pop(0)
            self.duration = self.data.pop(0)
            buf.write(struct.pack('=b3xhh', self.percent, self.pitch, self.duration))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, class_id, feedback_id, len, data):
        self = cls.__new__(cls)
        self.class_id = class_id
        self.feedback_id = feedback_id
        self.len = len
        self.data = data
        return self