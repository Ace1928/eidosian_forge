import xcffib
import struct
import io
def GetViewPort(self, screen, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xH2x', screen))
    return self.send_request(11, buf, GetViewPortCookie, is_checked=is_checked)