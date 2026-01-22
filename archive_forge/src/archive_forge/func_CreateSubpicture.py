import xcffib
import struct
import io
from . import xv
def CreateSubpicture(self, subpicture_id, context, xvimage_id, width, height, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIHH', subpicture_id, context, xvimage_id, width, height))
    return self.send_request(6, buf, CreateSubpictureCookie, is_checked=is_checked)