import xcffib
import struct
import io
def AuthConnection(self, screen, magic, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', screen, magic))
    return self.send_request(11, buf, AuthConnectionCookie, is_checked=is_checked)