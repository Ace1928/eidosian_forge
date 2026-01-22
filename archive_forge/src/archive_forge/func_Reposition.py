from pidWxDc import PiddleWxDc
from wxPython.wx import *
def Reposition(self):
    field = self.GetFieldRect(0)
    self.quitButton.SetPosition(wxPoint(field.x, field.y))
    self.quitButton.SetSize(wxSize(field.width / 2, field.height))
    self.clearButton.SetPosition(wxPoint(field.x + field.width / 2, field.y))
    self.clearButton.SetSize(wxSize(field.width / 2, field.height))
    field = self.GetFieldRect(1)
    self.click.SetPosition(wxPoint(field.x + field.width / 2 - 20, field.y))
    self.click.SetSize(wxSize(100, field.height))
    self.sizeChanged = false