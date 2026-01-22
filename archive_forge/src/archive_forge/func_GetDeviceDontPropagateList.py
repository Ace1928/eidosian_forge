import xcffib
import struct
import io
from . import xfixes
from . import xproto
def GetDeviceDontPropagateList(self, window, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(9, buf, GetDeviceDontPropagateListCookie, is_checked=is_checked)