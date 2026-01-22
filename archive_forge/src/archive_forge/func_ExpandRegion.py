import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def ExpandRegion(self, source, destination, left, right, top, bottom, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIHHHH', source, destination, left, right, top, bottom))
    return self.send_request(28, buf, is_checked=is_checked)