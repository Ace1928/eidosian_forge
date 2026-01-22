import xcffib
import struct
import io
from . import xproto
def GetDoublev(self, context_tag, pname, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', context_tag, pname))
    return self.send_request(114, buf, is_checked=is_checked)