import xcffib
import struct
import io
from . import xproto
def GetDeviceContext(self, device, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', device))
    return self.send_request(4, buf, GetDeviceContextCookie, is_checked=is_checked)