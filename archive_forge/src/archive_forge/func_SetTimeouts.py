import xcffib
import struct
import io
from . import xproto
def SetTimeouts(self, standby_timeout, suspend_timeout, off_timeout, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xHHH', standby_timeout, suspend_timeout, off_timeout))
    return self.send_request(3, buf, is_checked=is_checked)