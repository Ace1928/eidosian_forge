import xcffib
import struct
import io
def FreePixmap(self, pixmap, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', pixmap))
    return self.send_request(54, buf, is_checked=is_checked)