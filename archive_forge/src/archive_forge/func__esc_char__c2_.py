import unicodedata
def _esc_char__c2_(self):
    self._seq([lambda: self._ch('n'), lambda: self._succeed('\n')])