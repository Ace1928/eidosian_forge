import xcffib
import struct
import io
from . import xproto
def CopyContext(self, src, dest, mask, src_context_tag, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIII', src, dest, mask, src_context_tag))
    return self.send_request(10, buf, is_checked=is_checked)