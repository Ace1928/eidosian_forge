from pidWxDc import PiddleWxDc
from wxPython.wx import *
def OnClick(self, x, y):
    self.text = repr(x) + ',' + repr(y)
    self.click.SetValue(true)
    self.redraw()