import xcffib
import struct
import io
from . import xproto
def WaitMSC(self, drawable, target_msc_hi, target_msc_lo, divisor_hi, divisor_lo, remainder_hi, remainder_lo, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIIIII', drawable, target_msc_hi, target_msc_lo, divisor_hi, divisor_lo, remainder_hi, remainder_lo))
    return self.send_request(10, buf, WaitMSCCookie, is_checked=is_checked)