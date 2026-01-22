import xcffib
import struct
import io
def GetDeviceInfo(self, screen, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', screen))
    return self.send_request(10, buf, GetDeviceInfoCookie, is_checked=is_checked)