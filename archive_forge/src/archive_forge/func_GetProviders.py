import xcffib
import struct
import io
from . import xproto
from . import render
def GetProviders(self, window, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(32, buf, GetProvidersCookie, is_checked=is_checked)