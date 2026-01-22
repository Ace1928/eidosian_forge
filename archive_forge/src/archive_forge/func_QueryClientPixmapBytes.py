import xcffib
import struct
import io
from . import xproto
def QueryClientPixmapBytes(self, xid, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', xid))
    return self.send_request(3, buf, QueryClientPixmapBytesCookie, is_checked=is_checked)