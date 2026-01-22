import xcffib
import struct
import io
from . import xproto
class dri2Extension(xcffib.Extension):

    def QueryVersion(self, major_version, minor_version, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', major_version, minor_version))
        return self.send_request(0, buf, QueryVersionCookie, is_checked=is_checked)

    def Connect(self, window, driver_type, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', window, driver_type))
        return self.send_request(1, buf, ConnectCookie, is_checked=is_checked)

    def Authenticate(self, window, magic, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', window, magic))
        return self.send_request(2, buf, AuthenticateCookie, is_checked=is_checked)

    def CreateDrawable(self, drawable, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', drawable))
        return self.send_request(3, buf, is_checked=is_checked)

    def DestroyDrawable(self, drawable, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', drawable))
        return self.send_request(4, buf, is_checked=is_checked)

    def GetBuffers(self, drawable, count, attachments_len, attachments, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', drawable, count))
        buf.write(xcffib.pack_list(attachments, 'I'))
        return self.send_request(5, buf, GetBuffersCookie, is_checked=is_checked)

    def CopyRegion(self, drawable, region, dest, src, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIII', drawable, region, dest, src))
        return self.send_request(6, buf, CopyRegionCookie, is_checked=is_checked)

    def GetBuffersWithFormat(self, drawable, count, attachments_len, attachments, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', drawable, count))
        buf.write(xcffib.pack_list(attachments, AttachFormat))
        return self.send_request(7, buf, GetBuffersWithFormatCookie, is_checked=is_checked)

    def SwapBuffers(self, drawable, target_msc_hi, target_msc_lo, divisor_hi, divisor_lo, remainder_hi, remainder_lo, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIIIII', drawable, target_msc_hi, target_msc_lo, divisor_hi, divisor_lo, remainder_hi, remainder_lo))
        return self.send_request(8, buf, SwapBuffersCookie, is_checked=is_checked)

    def GetMSC(self, drawable, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', drawable))
        return self.send_request(9, buf, GetMSCCookie, is_checked=is_checked)

    def WaitMSC(self, drawable, target_msc_hi, target_msc_lo, divisor_hi, divisor_lo, remainder_hi, remainder_lo, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIIIII', drawable, target_msc_hi, target_msc_lo, divisor_hi, divisor_lo, remainder_hi, remainder_lo))
        return self.send_request(10, buf, WaitMSCCookie, is_checked=is_checked)

    def WaitSBC(self, drawable, target_sbc_hi, target_sbc_lo, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', drawable, target_sbc_hi, target_sbc_lo))
        return self.send_request(11, buf, WaitSBCCookie, is_checked=is_checked)

    def SwapInterval(self, drawable, interval, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', drawable, interval))
        return self.send_request(12, buf, is_checked=is_checked)

    def GetParam(self, drawable, param, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', drawable, param))
        return self.send_request(13, buf, GetParamCookie, is_checked=is_checked)