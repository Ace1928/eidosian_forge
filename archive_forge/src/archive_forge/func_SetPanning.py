import xcffib
import struct
import io
from . import xproto
from . import render
def SetPanning(self, crtc, timestamp, left, top, width, height, track_left, track_top, track_width, track_height, border_left, border_top, border_right, border_bottom, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIHHHHHHHHhhhh', crtc, timestamp, left, top, width, height, track_left, track_top, track_width, track_height, border_left, border_top, border_right, border_bottom))
    return self.send_request(29, buf, SetPanningCookie, is_checked=is_checked)