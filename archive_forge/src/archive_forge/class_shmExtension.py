import xcffib
import struct
import io
from . import xproto
class shmExtension(xcffib.Extension):

    def QueryVersion(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(0, buf, QueryVersionCookie, is_checked=is_checked)

    def Attach(self, shmseg, shmid, read_only, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIB3x', shmseg, shmid, read_only))
        return self.send_request(1, buf, is_checked=is_checked)

    def Detach(self, shmseg, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', shmseg))
        return self.send_request(2, buf, is_checked=is_checked)

    def PutImage(self, drawable, gc, total_width, total_height, src_x, src_y, src_width, src_height, dst_x, dst_y, depth, format, send_event, shmseg, offset, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIHHHHHHhhBBBxII', drawable, gc, total_width, total_height, src_x, src_y, src_width, src_height, dst_x, dst_y, depth, format, send_event, shmseg, offset))
        return self.send_request(3, buf, is_checked=is_checked)

    def GetImage(self, drawable, x, y, width, height, plane_mask, format, shmseg, offset, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIhhHHIB3xII', drawable, x, y, width, height, plane_mask, format, shmseg, offset))
        return self.send_request(4, buf, GetImageCookie, is_checked=is_checked)

    def CreatePixmap(self, pid, drawable, width, height, depth, shmseg, offset, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIHHB3xII', pid, drawable, width, height, depth, shmseg, offset))
        return self.send_request(5, buf, is_checked=is_checked)

    def AttachFd(self, shmseg, read_only, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIB3x', shmseg, read_only))
        return self.send_request(6, buf, is_checked=is_checked)

    def CreateSegment(self, shmseg, size, read_only, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIB3x', shmseg, size, read_only))
        return self.send_request(7, buf, CreateSegmentCookie, is_checked=is_checked)