from pidWxDc import PiddleWxDc
from wxPython.wx import *
def OnLeaveWindow(self):
    self.click.SetValue(false)
    self.text = ''
    self.redraw()