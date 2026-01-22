import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def IntersectRegion(self, source1, source2, destination, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', source1, source2, destination))
    return self.send_request(14, buf, is_checked=is_checked)