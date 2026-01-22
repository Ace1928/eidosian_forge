import xcffib
import struct
import io
from . import xproto
def IsDirect(self, context, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', context))
    return self.send_request(6, buf, IsDirectCookie, is_checked=is_checked)