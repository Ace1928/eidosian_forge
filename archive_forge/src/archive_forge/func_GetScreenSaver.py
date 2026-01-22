import xcffib
import struct
import io
def GetScreenSaver(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(108, buf, GetScreenSaverCookie, is_checked=is_checked)