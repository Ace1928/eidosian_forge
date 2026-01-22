import xcffib
import struct
import io
from . import xproto
from . import xfixes
class damageExtension(xcffib.Extension):

    def QueryVersion(self, client_major_version, client_minor_version, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', client_major_version, client_minor_version))
        return self.send_request(0, buf, QueryVersionCookie, is_checked=is_checked)

    def Create(self, damage, drawable, level, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIB3x', damage, drawable, level))
        return self.send_request(1, buf, is_checked=is_checked)

    def Destroy(self, damage, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', damage))
        return self.send_request(2, buf, is_checked=is_checked)

    def Subtract(self, damage, repair, parts, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', damage, repair, parts))
        return self.send_request(3, buf, is_checked=is_checked)

    def Add(self, drawable, region, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', drawable, region))
        return self.send_request(4, buf, is_checked=is_checked)