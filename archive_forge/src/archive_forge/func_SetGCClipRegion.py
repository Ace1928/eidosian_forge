import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def SetGCClipRegion(self, gc, region, x_origin, y_origin, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIhh', gc, region, x_origin, y_origin))
    return self.send_request(20, buf, is_checked=is_checked)