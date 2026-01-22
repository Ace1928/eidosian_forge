import xcffib
import struct
import io
from . import xproto
from . import render
def QueryOutputProperty(self, output, property, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', output, property))
    return self.send_request(11, buf, QueryOutputPropertyCookie, is_checked=is_checked)