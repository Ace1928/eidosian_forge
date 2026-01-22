import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def DeletePointerBarrier(self, barrier, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', barrier))
    return self.send_request(32, buf, is_checked=is_checked)