import xcffib
import struct
import io
from . import xproto
def SelectBuffer(self, context_tag, size, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIi', context_tag, size))
    return self.send_request(106, buf, is_checked=is_checked)