import unicodedata
def _esc_char__c9_(self):
    self._push('esc_char__c9')
    self._seq([self._esc_char__c9__s0_, lambda: self._bind(self._anything_, 'c'), lambda: self._succeed(self._get('c'))])
    self._pop('esc_char__c9')