import xcffib
import struct
import io
from . import xfixes
from . import xproto
def SelectExtensionEvent(self, window, num_classes, classes, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIH2x', window, num_classes))
    buf.write(xcffib.pack_list(classes, 'I'))
    return self.send_request(6, buf, is_checked=is_checked)