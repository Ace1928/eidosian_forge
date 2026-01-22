import xcffib
import struct
import io
def ChangeWindowAttributes(self, window, value_mask, value_list, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', window, value_mask))
    if value_mask & CW.BackPixmap:
        background_pixmap = value_list.pop(0)
        buf.write(struct.pack('=I', background_pixmap))
    if value_mask & CW.BackPixel:
        background_pixel = value_list.pop(0)
        buf.write(struct.pack('=I', background_pixel))
    if value_mask & CW.BorderPixmap:
        border_pixmap = value_list.pop(0)
        buf.write(struct.pack('=I', border_pixmap))
    if value_mask & CW.BorderPixel:
        border_pixel = value_list.pop(0)
        buf.write(struct.pack('=I', border_pixel))
    if value_mask & CW.BitGravity:
        bit_gravity = value_list.pop(0)
        buf.write(struct.pack('=I', bit_gravity))
    if value_mask & CW.WinGravity:
        win_gravity = value_list.pop(0)
        buf.write(struct.pack('=I', win_gravity))
    if value_mask & CW.BackingStore:
        backing_store = value_list.pop(0)
        buf.write(struct.pack('=I', backing_store))
    if value_mask & CW.BackingPlanes:
        backing_planes = value_list.pop(0)
        buf.write(struct.pack('=I', backing_planes))
    if value_mask & CW.BackingPixel:
        backing_pixel = value_list.pop(0)
        buf.write(struct.pack('=I', backing_pixel))
    if value_mask & CW.OverrideRedirect:
        override_redirect = value_list.pop(0)
        buf.write(struct.pack('=I', override_redirect))
    if value_mask & CW.SaveUnder:
        save_under = value_list.pop(0)
        buf.write(struct.pack('=I', save_under))
    if value_mask & CW.EventMask:
        event_mask = value_list.pop(0)
        buf.write(struct.pack('=I', event_mask))
    if value_mask & CW.DontPropagate:
        do_not_propogate_mask = value_list.pop(0)
        buf.write(struct.pack('=I', do_not_propogate_mask))
    if value_mask & CW.Colormap:
        colormap = value_list.pop(0)
        buf.write(struct.pack('=I', colormap))
    if value_mask & CW.Cursor:
        cursor = value_list.pop(0)
        buf.write(struct.pack('=I', cursor))
    return self.send_request(2, buf, is_checked=is_checked)