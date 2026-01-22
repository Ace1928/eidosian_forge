import wx
def SetSpellChecker(self, chkr):
    """Set the spell checker, advancing to the first error.
        Return True if error(s) to correct, else False."""
    self._checker = chkr
    return self.Advance()