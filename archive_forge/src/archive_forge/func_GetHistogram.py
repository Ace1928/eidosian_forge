import xcffib
import struct
import io
from . import xproto
def GetHistogram(self, context_tag, target, format, type, swap_bytes, reset, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIIBB', context_tag, target, format, type, swap_bytes, reset))
    return self.send_request(154, buf, GetHistogramCookie, is_checked=is_checked)