import xcffib
import struct
import io
def QueryTree(self, window, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(15, buf, QueryTreeCookie, is_checked=is_checked)