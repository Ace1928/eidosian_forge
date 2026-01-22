import xcffib
import struct
import io
from . import xproto
def WaitSBC(self, drawable, target_sbc_hi, target_sbc_lo, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', drawable, target_sbc_hi, target_sbc_lo))
    return self.send_request(11, buf, WaitSBCCookie, is_checked=is_checked)