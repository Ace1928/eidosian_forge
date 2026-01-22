import xcffib
import struct
import io
from . import xproto
def GetState(self, window, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(1, buf, GetStateCookie, is_checked=is_checked)