import xcffib
import struct
import io
from . import xproto
def SelectInput(self, event_mask, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', event_mask))
    return self.send_request(8, buf, is_checked=is_checked)