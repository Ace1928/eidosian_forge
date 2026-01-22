import xcffib
import struct
import io
from . import xproto
def QueryScreens(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(5, buf, QueryScreensCookie, is_checked=is_checked)