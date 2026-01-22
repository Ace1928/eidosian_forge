import xcffib
import struct
import io
from . import xproto
from . import render
def SetOutputPrimary(self, window, output, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', window, output))
    return self.send_request(30, buf, is_checked=is_checked)