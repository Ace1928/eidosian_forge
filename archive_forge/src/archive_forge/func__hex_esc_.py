import unicodedata
def _hex_esc_(self):
    self._push('hex_esc')
    self._seq([lambda: self._ch('x'), lambda: self._bind(self._hex_, 'h1'), lambda: self._bind(self._hex_, 'h2'), lambda: self._succeed(self._xtou(self._get('h1') + self._get('h2')))])
    self._pop('hex_esc')