import xcffib
import struct
import io
def SetGammaRamp(self, screen, size, red, green, blue, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xHH', screen, size))
    buf.write(xcffib.pack_list(red, 'H'))
    buf.write(xcffib.pack_list(green, 'H'))
    buf.write(xcffib.pack_list(blue, 'H'))
    return self.send_request(18, buf, is_checked=is_checked)