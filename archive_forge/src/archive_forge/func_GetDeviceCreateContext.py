import xcffib
import struct
import io
from . import xproto
def GetDeviceCreateContext(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(2, buf, GetDeviceCreateContextCookie, is_checked=is_checked)