import xcffib
import struct
import io
from . import xproto
def QueryContext(self, context, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', context))
    return self.send_request(25, buf, QueryContextCookie, is_checked=is_checked)