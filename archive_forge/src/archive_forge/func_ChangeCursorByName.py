import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def ChangeCursorByName(self, src, nbytes, name, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIH2x', src, nbytes))
    buf.write(xcffib.pack_list(name, 'c'))
    return self.send_request(27, buf, is_checked=is_checked)