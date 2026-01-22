import xcffib
import struct
import io
from . import xproto
class resExtension(xcffib.Extension):

    def QueryVersion(self, client_major, client_minor, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBB', client_major, client_minor))
        return self.send_request(0, buf, QueryVersionCookie, is_checked=is_checked)

    def QueryClients(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(1, buf, QueryClientsCookie, is_checked=is_checked)

    def QueryClientResources(self, xid, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', xid))
        return self.send_request(2, buf, QueryClientResourcesCookie, is_checked=is_checked)

    def QueryClientPixmapBytes(self, xid, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', xid))
        return self.send_request(3, buf, QueryClientPixmapBytesCookie, is_checked=is_checked)

    def QueryClientIds(self, num_specs, specs, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', num_specs))
        buf.write(xcffib.pack_list(specs, ClientIdSpec))
        return self.send_request(4, buf, QueryClientIdsCookie, is_checked=is_checked)

    def QueryResourceBytes(self, client, num_specs, specs, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', client, num_specs))
        buf.write(xcffib.pack_list(specs, ResourceIdSpec))
        return self.send_request(5, buf, QueryResourceBytesCookie, is_checked=is_checked)