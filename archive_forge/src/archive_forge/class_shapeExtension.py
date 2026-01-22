import xcffib
import struct
import io
from . import xproto
class shapeExtension(xcffib.Extension):

    def QueryVersion(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(0, buf, QueryVersionCookie, is_checked=is_checked)

    def Rectangles(self, operation, destination_kind, ordering, destination_window, x_offset, y_offset, rectangles_len, rectangles, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBBBxIhh', operation, destination_kind, ordering, destination_window, x_offset, y_offset))
        buf.write(xcffib.pack_list(rectangles, xproto.RECTANGLE))
        return self.send_request(1, buf, is_checked=is_checked)

    def Mask(self, operation, destination_kind, destination_window, x_offset, y_offset, source_bitmap, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBB2xIhhI', operation, destination_kind, destination_window, x_offset, y_offset, source_bitmap))
        return self.send_request(2, buf, is_checked=is_checked)

    def Combine(self, operation, destination_kind, source_kind, destination_window, x_offset, y_offset, source_window, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBBBxIhhI', operation, destination_kind, source_kind, destination_window, x_offset, y_offset, source_window))
        return self.send_request(3, buf, is_checked=is_checked)

    def Offset(self, destination_kind, destination_window, x_offset, y_offset, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xB3xIhh', destination_kind, destination_window, x_offset, y_offset))
        return self.send_request(4, buf, is_checked=is_checked)

    def QueryExtents(self, destination_window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', destination_window))
        return self.send_request(5, buf, QueryExtentsCookie, is_checked=is_checked)

    def SelectInput(self, destination_window, enable, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIB3x', destination_window, enable))
        return self.send_request(6, buf, is_checked=is_checked)

    def InputSelected(self, destination_window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', destination_window))
        return self.send_request(7, buf, InputSelectedCookie, is_checked=is_checked)

    def GetRectangles(self, window, source_kind, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIB3x', window, source_kind))
        return self.send_request(8, buf, GetRectanglesCookie, is_checked=is_checked)