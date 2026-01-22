import xcffib
import struct
import io
from . import xproto
class xtestExtension(xcffib.Extension):

    def GetVersion(self, major_version, minor_version, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBxH', major_version, minor_version))
        return self.send_request(0, buf, GetVersionCookie, is_checked=is_checked)

    def CompareCursor(self, window, cursor, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', window, cursor))
        return self.send_request(1, buf, CompareCursorCookie, is_checked=is_checked)

    def FakeInput(self, type, detail, time, root, rootX, rootY, deviceid, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBB2xII8xhh7xB', type, detail, time, root, rootX, rootY, deviceid))
        return self.send_request(2, buf, is_checked=is_checked)

    def GrabControl(self, impervious, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xB3x', impervious))
        return self.send_request(3, buf, is_checked=is_checked)