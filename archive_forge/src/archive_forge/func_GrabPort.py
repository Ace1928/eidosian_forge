import xcffib
import struct
import io
from . import xproto
from . import shm
def GrabPort(self, port, time, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', port, time))
    return self.send_request(3, buf, GrabPortCookie, is_checked=is_checked)