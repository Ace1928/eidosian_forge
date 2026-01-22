import xcffib
import struct
import io
def GrabKey(self, owner_events, grab_window, modifiers, key, pointer_mode, keyboard_mode, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xIHBBB3x', owner_events, grab_window, modifiers, key, pointer_mode, keyboard_mode))
    return self.send_request(33, buf, is_checked=is_checked)