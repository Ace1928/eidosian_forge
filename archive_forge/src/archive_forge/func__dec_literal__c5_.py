import unicodedata
def _dec_literal__c5_(self):
    self._push('dec_literal__c5')
    self._seq([lambda: self._bind(self._frac_, 'f'), lambda: self._succeed(self._get('f'))])
    self._pop('dec_literal__c5')