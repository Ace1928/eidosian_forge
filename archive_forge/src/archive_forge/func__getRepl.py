import gtk
def _getRepl(self):
    """Get the chosen replacement string."""
    repl = self.replace_text.get_text()
    repl = self._checker.coerce_string(repl)
    return repl