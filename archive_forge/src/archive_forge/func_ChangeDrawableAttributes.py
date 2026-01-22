import xcffib
import struct
import io
from . import xproto
def ChangeDrawableAttributes(self, drawable, num_attribs, attribs, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', drawable, num_attribs))
    buf.write(xcffib.pack_list(attribs, 'I'))
    return self.send_request(30, buf, is_checked=is_checked)