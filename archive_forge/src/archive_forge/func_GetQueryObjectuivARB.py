import xcffib
import struct
import io
from . import xproto
def GetQueryObjectuivARB(self, context_tag, id, pname, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', context_tag, id, pname))
    return self.send_request(166, buf, GetQueryObjectuivARBCookie, is_checked=is_checked)