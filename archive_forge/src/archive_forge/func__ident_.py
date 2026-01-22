import unicodedata
def _ident_(self):
    self._push('ident')
    self._seq([lambda: self._bind(self._id_start_, 'hd'), self._ident__s1_, lambda: self._succeed(self._join('', [self._get('hd')] + self._get('tl')))])
    self._pop('ident')