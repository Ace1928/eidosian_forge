import unicodedata
def _opt(self, rule):
    p = self.pos
    rule()
    if self.failed:
        self._succeed([], p)
    else:
        self._succeed([self.val])