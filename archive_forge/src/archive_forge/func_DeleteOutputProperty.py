import xcffib
import struct
import io
from . import xproto
from . import render
def DeleteOutputProperty(self, output, property, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', output, property))
    return self.send_request(14, buf, is_checked=is_checked)