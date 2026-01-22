import xcffib
import struct
import io
from . import xproto
def GetDrawableAttributes(self, drawable, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', drawable))
    return self.send_request(29, buf, GetDrawableAttributesCookie, is_checked=is_checked)