import xcffib
import struct
import io
from . import xproto
from . import xfixes
class compositeExtension(xcffib.Extension):

    def QueryVersion(self, client_major_version, client_minor_version, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', client_major_version, client_minor_version))
        return self.send_request(0, buf, QueryVersionCookie, is_checked=is_checked)

    def RedirectWindow(self, window, update, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIB3x', window, update))
        return self.send_request(1, buf, is_checked=is_checked)

    def RedirectSubwindows(self, window, update, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIB3x', window, update))
        return self.send_request(2, buf, is_checked=is_checked)

    def UnredirectWindow(self, window, update, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIB3x', window, update))
        return self.send_request(3, buf, is_checked=is_checked)

    def UnredirectSubwindows(self, window, update, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIB3x', window, update))
        return self.send_request(4, buf, is_checked=is_checked)

    def CreateRegionFromBorderClip(self, region, window, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', region, window))
        return self.send_request(5, buf, is_checked=is_checked)

    def NameWindowPixmap(self, window, pixmap, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', window, pixmap))
        return self.send_request(6, buf, is_checked=is_checked)

    def GetOverlayWindow(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(7, buf, GetOverlayWindowCookie, is_checked=is_checked)

    def ReleaseOverlayWindow(self, window, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(8, buf, is_checked=is_checked)