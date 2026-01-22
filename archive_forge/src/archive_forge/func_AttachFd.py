import xcffib
import struct
import io
from . import xproto
def AttachFd(self, shmseg, read_only, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIB3x', shmseg, read_only))
    return self.send_request(6, buf, is_checked=is_checked)