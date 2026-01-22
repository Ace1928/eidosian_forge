import xcffib
import struct
import io
def ConfigureWindow(self, window, value_mask, value_list, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIH2x', window, value_mask))
    if value_mask & ConfigWindow.X:
        x = value_list.pop(0)
        buf.write(struct.pack('=i', x))
    if value_mask & ConfigWindow.Y:
        y = value_list.pop(0)
        buf.write(struct.pack('=i', y))
    if value_mask & ConfigWindow.Width:
        width = value_list.pop(0)
        buf.write(struct.pack('=I', width))
    if value_mask & ConfigWindow.Height:
        height = value_list.pop(0)
        buf.write(struct.pack('=I', height))
    if value_mask & ConfigWindow.BorderWidth:
        border_width = value_list.pop(0)
        buf.write(struct.pack('=I', border_width))
    if value_mask & ConfigWindow.Sibling:
        sibling = value_list.pop(0)
        buf.write(struct.pack('=I', sibling))
    if value_mask & ConfigWindow.StackMode:
        stack_mode = value_list.pop(0)
        buf.write(struct.pack('=I', stack_mode))
    return self.send_request(12, buf, is_checked=is_checked)