import gtk
def _onIgnoreAll(self, w, *args):
    print(['ignore all'])
    self._checker.ignore_always()
    self._advance()