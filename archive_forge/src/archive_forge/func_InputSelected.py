import xcffib
import struct
import io
from . import xproto
def InputSelected(self, destination_window, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', destination_window))
    return self.send_request(7, buf, InputSelectedCookie, is_checked=is_checked)