import xcffib
import struct
import io
def GetClientDriverName(self, screen, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', screen))
    return self.send_request(4, buf, GetClientDriverNameCookie, is_checked=is_checked)