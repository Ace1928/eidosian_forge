from pidWxDc import PiddleWxDc
from wxPython.wx import *
def _OnKey(self, event):
    code = event.KeyCode()
    key = None
    if code >= 0 and code < 256:
        key = chr(event.KeyCode())
    modifier = []
    if event.ControlDown():
        modifier.append('modControl')
    if event.ShiftDown():
        modifier.append('modshift')
    self.onKey(self, key, tuple(modifier))