import xcffib
import struct
import io
from . import xproto
def WaitX(self, context_tag, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', context_tag))
    return self.send_request(9, buf, is_checked=is_checked)