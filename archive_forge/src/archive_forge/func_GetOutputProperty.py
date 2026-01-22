import xcffib
import struct
import io
from . import xproto
from . import render
def GetOutputProperty(self, output, property, type, long_offset, long_length, delete, pending, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIIIBB2x', output, property, type, long_offset, long_length, delete, pending))
    return self.send_request(15, buf, GetOutputPropertyCookie, is_checked=is_checked)