import xcffib
import struct
import io
from . import xproto
def PixelStorei(self, context_tag, pname, datum, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIi', context_tag, pname, datum))
    return self.send_request(110, buf, is_checked=is_checked)