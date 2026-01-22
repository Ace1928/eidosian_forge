import xcffib
import struct
import io
from . import xproto
from . import render
def FreeLease(self, lid, terminate, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIB', lid, terminate))
    return self.send_request(46, buf, is_checked=is_checked)