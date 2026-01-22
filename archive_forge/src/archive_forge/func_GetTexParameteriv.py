import xcffib
import struct
import io
from . import xproto
def GetTexParameteriv(self, context_tag, target, pname, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', context_tag, target, pname))
    return self.send_request(137, buf, GetTexParameterivCookie, is_checked=is_checked)