import xcffib
import struct
import io
from . import xproto
def GetClipPlane(self, context_tag, plane, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIi', context_tag, plane))
    return self.send_request(113, buf, is_checked=is_checked)