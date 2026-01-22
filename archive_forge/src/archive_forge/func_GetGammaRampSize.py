import xcffib
import struct
import io
def GetGammaRampSize(self, screen, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xH2x', screen))
    return self.send_request(19, buf, GetGammaRampSizeCookie, is_checked=is_checked)