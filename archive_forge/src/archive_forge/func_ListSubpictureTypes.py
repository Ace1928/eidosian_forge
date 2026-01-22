import xcffib
import struct
import io
from . import xv
def ListSubpictureTypes(self, port_id, surface_id, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', port_id, surface_id))
    return self.send_request(8, buf, ListSubpictureTypesCookie, is_checked=is_checked)