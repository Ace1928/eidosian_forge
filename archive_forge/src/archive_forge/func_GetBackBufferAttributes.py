import xcffib
import struct
import io
from . import xproto
def GetBackBufferAttributes(self, buffer, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', buffer))
    return self.send_request(7, buf, GetBackBufferAttributesCookie, is_checked=is_checked)