import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def SetClientDisconnectMode(self, disconnect_mode, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', disconnect_mode))
    return self.send_request(33, buf, is_checked=is_checked)