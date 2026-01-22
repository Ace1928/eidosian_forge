from pidWxDc import PiddleWxDc
from wxPython.wx import *
def OnIdle(self, evt):
    if self.sizeChanged:
        self.Reposition()