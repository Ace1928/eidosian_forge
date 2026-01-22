import xcffib
import struct
import io
from . import xproto
def SetAttributes(self, drawable, x, y, width, height, border_width, _class, depth, visual, value_mask, value_list, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIhhHHHBBII', drawable, x, y, width, height, border_width, _class, depth, visual, value_mask))
    if value_mask & xproto.CW:
        background_pixmap = value_list.pop(0)
        buf.write(struct.pack('=I', background_pixmap))
    if value_mask & xproto.CW:
        background_pixel = value_list.pop(0)
        buf.write(struct.pack('=I', background_pixel))
    if value_mask & xproto.CW:
        border_pixmap = value_list.pop(0)
        buf.write(struct.pack('=I', border_pixmap))
    if value_mask & xproto.CW:
        border_pixel = value_list.pop(0)
        buf.write(struct.pack('=I', border_pixel))
    if value_mask & xproto.CW:
        bit_gravity = value_list.pop(0)
        buf.write(struct.pack('=I', bit_gravity))
    if value_mask & xproto.CW:
        win_gravity = value_list.pop(0)
        buf.write(struct.pack('=I', win_gravity))
    if value_mask & xproto.CW:
        backing_store = value_list.pop(0)
        buf.write(struct.pack('=I', backing_store))
    if value_mask & xproto.CW:
        backing_planes = value_list.pop(0)
        buf.write(struct.pack('=I', backing_planes))
    if value_mask & xproto.CW:
        backing_pixel = value_list.pop(0)
        buf.write(struct.pack('=I', backing_pixel))
    if value_mask & xproto.CW:
        override_redirect = value_list.pop(0)
        buf.write(struct.pack('=I', override_redirect))
    if value_mask & xproto.CW:
        save_under = value_list.pop(0)
        buf.write(struct.pack('=I', save_under))
    if value_mask & xproto.CW:
        event_mask = value_list.pop(0)
        buf.write(struct.pack('=I', event_mask))
    if value_mask & xproto.CW:
        do_not_propogate_mask = value_list.pop(0)
        buf.write(struct.pack('=I', do_not_propogate_mask))
    if value_mask & xproto.CW:
        colormap = value_list.pop(0)
        buf.write(struct.pack('=I', colormap))
    if value_mask & xproto.CW:
        cursor = value_list.pop(0)
        buf.write(struct.pack('=I', cursor))
    return self.send_request(3, buf, is_checked=is_checked)