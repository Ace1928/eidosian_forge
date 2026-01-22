import xcffib
import struct
import io
from . import xproto
def QueryFence(self, fence, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', fence))
    return self.send_request(18, buf, QueryFenceCookie, is_checked=is_checked)