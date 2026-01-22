import xcffib
import struct
import io
from . import xproto
def GetTexGendv(self, context_tag, coord, pname, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', context_tag, coord, pname))
    return self.send_request(132, buf, is_checked=is_checked)