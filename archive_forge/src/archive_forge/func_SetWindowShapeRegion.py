import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def SetWindowShapeRegion(self, dest, dest_kind, x_offset, y_offset, region, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIB3xhhI', dest, dest_kind, x_offset, y_offset, region))
    return self.send_request(21, buf, is_checked=is_checked)