import xcffib
import struct
import io
from . import xproto
from . import render
def ListOutputProperties(self, output, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', output))
    return self.send_request(10, buf, ListOutputPropertiesCookie, is_checked=is_checked)