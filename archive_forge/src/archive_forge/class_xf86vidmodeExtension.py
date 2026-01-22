import xcffib
import struct
import io
class xf86vidmodeExtension(xcffib.Extension):

    def QueryVersion(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(0, buf, QueryVersionCookie, is_checked=is_checked)

    def GetModeLine(self, screen, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xH2x', screen))
        return self.send_request(1, buf, GetModeLineCookie, is_checked=is_checked)

    def ModModeLine(self, screen, hdisplay, hsyncstart, hsyncend, htotal, hskew, vdisplay, vsyncstart, vsyncend, vtotal, flags, privsize, private, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIHHHHHHHHH2xI12xI', screen, hdisplay, hsyncstart, hsyncend, htotal, hskew, vdisplay, vsyncstart, vsyncend, vtotal, flags, privsize))
        buf.write(xcffib.pack_list(private, 'B'))
        return self.send_request(2, buf, is_checked=is_checked)

    def SwitchMode(self, screen, zoom, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xHH', screen, zoom))
        return self.send_request(3, buf, is_checked=is_checked)

    def GetMonitor(self, screen, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xH2x', screen))
        return self.send_request(4, buf, GetMonitorCookie, is_checked=is_checked)

    def LockModeSwitch(self, screen, lock, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xHH', screen, lock))
        return self.send_request(5, buf, is_checked=is_checked)

    def GetAllModeLines(self, screen, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xH2x', screen))
        return self.send_request(6, buf, GetAllModeLinesCookie, is_checked=is_checked)

    def AddModeLine(self, screen, dotclock, hdisplay, hsyncstart, hsyncend, htotal, hskew, vdisplay, vsyncstart, vsyncend, vtotal, flags, privsize, after_dotclock, after_hdisplay, after_hsyncstart, after_hsyncend, after_htotal, after_hskew, after_vdisplay, after_vsyncstart, after_vsyncend, after_vtotal, after_flags, private, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIHHHHHHHHH2xI12xIIHHHHHHHHH2xI12x', screen, dotclock, hdisplay, hsyncstart, hsyncend, htotal, hskew, vdisplay, vsyncstart, vsyncend, vtotal, flags, privsize, after_dotclock, after_hdisplay, after_hsyncstart, after_hsyncend, after_htotal, after_hskew, after_vdisplay, after_vsyncstart, after_vsyncend, after_vtotal, after_flags))
        buf.write(xcffib.pack_list(private, 'B'))
        return self.send_request(7, buf, is_checked=is_checked)

    def DeleteModeLine(self, screen, dotclock, hdisplay, hsyncstart, hsyncend, htotal, hskew, vdisplay, vsyncstart, vsyncend, vtotal, flags, privsize, private, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIHHHHHHHHH2xI12xI', screen, dotclock, hdisplay, hsyncstart, hsyncend, htotal, hskew, vdisplay, vsyncstart, vsyncend, vtotal, flags, privsize))
        buf.write(xcffib.pack_list(private, 'B'))
        return self.send_request(8, buf, is_checked=is_checked)

    def ValidateModeLine(self, screen, dotclock, hdisplay, hsyncstart, hsyncend, htotal, hskew, vdisplay, vsyncstart, vsyncend, vtotal, flags, privsize, private, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIHHHHHHHHH2xI12xI', screen, dotclock, hdisplay, hsyncstart, hsyncend, htotal, hskew, vdisplay, vsyncstart, vsyncend, vtotal, flags, privsize))
        buf.write(xcffib.pack_list(private, 'B'))
        return self.send_request(9, buf, ValidateModeLineCookie, is_checked=is_checked)

    def SwitchToMode(self, screen, dotclock, hdisplay, hsyncstart, hsyncend, htotal, hskew, vdisplay, vsyncstart, vsyncend, vtotal, flags, privsize, private, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIHHHHHHHHH2xI12xI', screen, dotclock, hdisplay, hsyncstart, hsyncend, htotal, hskew, vdisplay, vsyncstart, vsyncend, vtotal, flags, privsize))
        buf.write(xcffib.pack_list(private, 'B'))
        return self.send_request(10, buf, is_checked=is_checked)

    def GetViewPort(self, screen, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xH2x', screen))
        return self.send_request(11, buf, GetViewPortCookie, is_checked=is_checked)

    def SetViewPort(self, screen, x, y, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xH2xII', screen, x, y))
        return self.send_request(12, buf, is_checked=is_checked)

    def GetDotClocks(self, screen, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xH2x', screen))
        return self.send_request(13, buf, GetDotClocksCookie, is_checked=is_checked)

    def SetClientVersion(self, major, minor, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xHH', major, minor))
        return self.send_request(14, buf, is_checked=is_checked)

    def SetGamma(self, screen, red, green, blue, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xH2xIII12x', screen, red, green, blue))
        return self.send_request(15, buf, is_checked=is_checked)

    def GetGamma(self, screen, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xH26x', screen))
        return self.send_request(16, buf, GetGammaCookie, is_checked=is_checked)

    def GetGammaRamp(self, screen, size, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xHH', screen, size))
        return self.send_request(17, buf, GetGammaRampCookie, is_checked=is_checked)

    def SetGammaRamp(self, screen, size, red, green, blue, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xHH', screen, size))
        buf.write(xcffib.pack_list(red, 'H'))
        buf.write(xcffib.pack_list(green, 'H'))
        buf.write(xcffib.pack_list(blue, 'H'))
        return self.send_request(18, buf, is_checked=is_checked)

    def GetGammaRampSize(self, screen, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xH2x', screen))
        return self.send_request(19, buf, GetGammaRampSizeCookie, is_checked=is_checked)

    def GetPermissions(self, screen, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xH2x', screen))
        return self.send_request(20, buf, GetPermissionsCookie, is_checked=is_checked)