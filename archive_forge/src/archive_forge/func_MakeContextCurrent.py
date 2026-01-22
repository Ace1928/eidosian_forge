import xcffib
import struct
import io
from . import xproto
def MakeContextCurrent(self, old_context_tag, drawable, read_drawable, context, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIII', old_context_tag, drawable, read_drawable, context))
    return self.send_request(26, buf, MakeContextCurrentCookie, is_checked=is_checked)