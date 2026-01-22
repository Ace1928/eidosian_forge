import xcffib
import struct
import io
from . import xproto
from . import shm
def ListImageFormats(self, port, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', port))
    return self.send_request(16, buf, ListImageFormatsCookie, is_checked=is_checked)