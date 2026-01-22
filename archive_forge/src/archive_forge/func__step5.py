def _step5(self):
    """Remove a final -e if _m() > 1, and change -ll to -l if m() > 1."""
    k = self.j = self.k
    if self.b[k] == 'e':
        a = self._m()
        if a > 1 or (a == 1 and (not self._cvc(k - 1))):
            self.k -= 1
    if self.b[self.k] == 'l' and self._doublec(self.k) and (self._m() > 1):
        self.k -= 1