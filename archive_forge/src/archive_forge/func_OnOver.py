from pidWxDc import PiddleWxDc
from wxPython.wx import *
def OnOver(self, x, y):
    self.text = repr(x) + ',' + repr(y)
    self.redraw()