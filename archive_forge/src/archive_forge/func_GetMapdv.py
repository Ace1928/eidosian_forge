import xcffib
import struct
import io
from . import xproto
def GetMapdv(self, context_tag, target, query, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', context_tag, target, query))
    return self.send_request(120, buf, is_checked=is_checked)