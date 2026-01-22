import xcffib
import struct
import io
def Start(self, screen, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', screen))
    return self.send_request(1, buf, StartCookie, is_checked=is_checked)