import xcffib
import struct
import io
from . import xproto
def GetColorTableParameteriv(self, context_tag, target, pname, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', context_tag, target, pname))
    return self.send_request(149, buf, GetColorTableParameterivCookie, is_checked=is_checked)