import xcffib
import struct
import io
from . import xproto
def Suspend(self, suspend, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', suspend))
    return self.send_request(5, buf, is_checked=is_checked)