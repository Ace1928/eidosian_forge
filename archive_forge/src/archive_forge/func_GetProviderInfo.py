import xcffib
import struct
import io
from . import xproto
from . import render
def GetProviderInfo(self, provider, config_timestamp, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', provider, config_timestamp))
    return self.send_request(33, buf, GetProviderInfoCookie, is_checked=is_checked)