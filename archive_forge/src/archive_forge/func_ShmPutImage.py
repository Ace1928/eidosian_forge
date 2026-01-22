import xcffib
import struct
import io
from . import xproto
from . import shm
def ShmPutImage(self, port, drawable, gc, shmseg, id, offset, src_x, src_y, src_w, src_h, drw_x, drw_y, drw_w, drw_h, width, height, send_event, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIIIIhhHHhhHHHHB3x', port, drawable, gc, shmseg, id, offset, src_x, src_y, src_w, src_h, drw_x, drw_y, drw_w, drw_h, width, height, send_event))
    return self.send_request(19, buf, is_checked=is_checked)