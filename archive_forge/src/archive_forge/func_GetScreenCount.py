import xcffib
import struct
import io
from . import xproto
def GetScreenCount(self, window, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(2, buf, GetScreenCountCookie, is_checked=is_checked)