import xcffib
import struct
import io
from . import xproto
def GetPolygonStipple(self, context_tag, lsb_first, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIB', context_tag, lsb_first))
    return self.send_request(128, buf, GetPolygonStippleCookie, is_checked=is_checked)