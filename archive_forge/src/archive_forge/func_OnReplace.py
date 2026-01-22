import wx
def OnReplace(self, evt):
    """Callback for the "replace" button."""
    repl = self.GetRepl()
    if repl:
        self._checker.replace(repl)
    self.Advance()