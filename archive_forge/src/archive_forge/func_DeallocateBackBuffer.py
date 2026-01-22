import xcffib
import struct
import io
from . import xproto
def DeallocateBackBuffer(self, buffer, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', buffer))
    return self.send_request(2, buf, is_checked=is_checked)