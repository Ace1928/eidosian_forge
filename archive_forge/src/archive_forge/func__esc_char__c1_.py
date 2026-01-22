import unicodedata
def _esc_char__c1_(self):
    self._seq([lambda: self._ch('f'), lambda: self._succeed('\x0c')])