import xcffib
import struct
import io
from . import xproto
def PixelStoref(self, context_tag, pname, datum, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIf', context_tag, pname, datum))
    return self.send_request(109, buf, is_checked=is_checked)