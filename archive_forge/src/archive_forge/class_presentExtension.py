import xcffib
import struct
import io
from . import xproto
from . import randr
from . import xfixes
from . import sync
class presentExtension(xcffib.Extension):

    def QueryVersion(self, major_version, minor_version, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', major_version, minor_version))
        return self.send_request(0, buf, QueryVersionCookie, is_checked=is_checked)

    def SelectInput(self, eid, window, event_mask, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', eid, window, event_mask))
        return self.send_request(3, buf, is_checked=is_checked)

    def QueryCapabilities(self, target, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', target))
        return self.send_request(4, buf, QueryCapabilitiesCookie, is_checked=is_checked)