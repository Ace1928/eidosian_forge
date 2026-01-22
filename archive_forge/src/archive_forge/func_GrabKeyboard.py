import xcffib
import struct
import io
def GrabKeyboard(self, owner_events, grab_window, time, pointer_mode, keyboard_mode, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xIIBB2x', owner_events, grab_window, time, pointer_mode, keyboard_mode))
    return self.send_request(31, buf, GrabKeyboardCookie, is_checked=is_checked)