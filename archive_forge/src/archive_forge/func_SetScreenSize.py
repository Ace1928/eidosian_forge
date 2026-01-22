import xcffib
import struct
import io
from . import xproto
from . import render
def SetScreenSize(self, window, width, height, mm_width, mm_height, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIHHII', window, width, height, mm_width, mm_height))
    return self.send_request(7, buf, is_checked=is_checked)