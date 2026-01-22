import xcffib
import struct
import io
from . import xproto
from . import shm
def QueryAdaptors(self, window, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(1, buf, QueryAdaptorsCookie, is_checked=is_checked)