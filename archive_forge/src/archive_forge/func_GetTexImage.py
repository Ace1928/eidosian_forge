import xcffib
import struct
import io
from . import xproto
def GetTexImage(self, context_tag, target, level, format, type, swap_bytes, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIiIIB', context_tag, target, level, format, type, swap_bytes))
    return self.send_request(135, buf, GetTexImageCookie, is_checked=is_checked)