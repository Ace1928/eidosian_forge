import xcffib
import struct
import io
from . import xproto
from . import render
def SetCrtcGamma(self, crtc, size, red, green, blue, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIH2x', crtc, size))
    buf.write(xcffib.pack_list(red, 'H'))
    buf.write(xcffib.pack_list(green, 'H'))
    buf.write(xcffib.pack_list(blue, 'H'))
    return self.send_request(24, buf, is_checked=is_checked)