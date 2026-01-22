import wx
def OnIgnore(self, evt):
    """Callback for the "ignore" button.
        This simply advances to the next error.
        """
    self.Advance()