import xcffib
import struct
import io
from . import xproto
def GenTextures(self, context_tag, n, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIi', context_tag, n))
    return self.send_request(145, buf, GenTexturesCookie, is_checked=is_checked)