import unicodedata
def _dec_literal__c1_(self):
    self._push('dec_literal__c1')
    self._seq([lambda: self._bind(self._dec_int_lit_, 'd'), lambda: self._bind(self._frac_, 'f'), lambda: self._succeed(self._get('d') + self._get('f'))])
    self._pop('dec_literal__c1')