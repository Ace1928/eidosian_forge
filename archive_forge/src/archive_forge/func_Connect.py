import xcffib
import struct
import io
from . import xproto
def Connect(self, window, driver_type, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', window, driver_type))
    return self.send_request(1, buf, ConnectCookie, is_checked=is_checked)