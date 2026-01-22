import unicodedata
def _esc_char__c12_(self):
    self._push('esc_char__c12')
    self._seq([lambda: self._bind(self._unicode_esc_, 'c'), lambda: self._succeed(self._get('c'))])
    self._pop('esc_char__c12')