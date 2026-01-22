import xcffib
import struct
import io
from . import xproto
def GetColorTable(self, context_tag, target, format, type, swap_bytes, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIIB', context_tag, target, format, type, swap_bytes))
    return self.send_request(147, buf, GetColorTableCookie, is_checked=is_checked)