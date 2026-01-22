import xcffib
import struct
import io
from . import xproto
def CreateSegment(self, shmseg, size, read_only, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIB3x', shmseg, size, read_only))
    return self.send_request(7, buf, CreateSegmentCookie, is_checked=is_checked)