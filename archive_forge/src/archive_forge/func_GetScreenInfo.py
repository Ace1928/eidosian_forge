import xcffib
import struct
import io
from . import xproto
from . import render
def GetScreenInfo(self, window, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(5, buf, GetScreenInfoCookie, is_checked=is_checked)