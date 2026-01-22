from pidWxDc import PiddleWxDc
from wxPython.wx import *
def _OnClick(self, event):
    if self.interactive == false:
        return None
    if event.GetY() <= self.size[1]:
        self.onClick(self, event.GetX(), event.GetY())