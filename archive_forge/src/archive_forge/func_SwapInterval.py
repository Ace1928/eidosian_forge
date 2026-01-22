import xcffib
import struct
import io
from . import xproto
def SwapInterval(self, drawable, interval, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', drawable, interval))
    return self.send_request(12, buf, is_checked=is_checked)