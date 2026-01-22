import xcffib
import struct
import io
def EnableContext(self, context, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', context))
    return self.send_request(5, buf, EnableContextCookie, is_checked=is_checked)