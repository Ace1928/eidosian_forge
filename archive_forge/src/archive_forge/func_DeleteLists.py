import xcffib
import struct
import io
from . import xproto
def DeleteLists(self, context_tag, list, range, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIi', context_tag, list, range))
    return self.send_request(103, buf, is_checked=is_checked)