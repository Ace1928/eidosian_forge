import unicodedata
def _esc_char__c4_(self):
    self._seq([lambda: self._ch('t'), lambda: self._succeed('\t')])