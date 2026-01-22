import xcffib
import struct
import io
def GetMonitor(self, screen, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xH2x', screen))
    return self.send_request(4, buf, GetMonitorCookie, is_checked=is_checked)