import unicodedata
def _dec_int_lit__c1_(self):
    self._push('dec_int_lit__c1')
    self._seq([lambda: self._bind(self._nonzerodigit_, 'd'), self._dec_int_lit__c1__s1_, lambda: self._succeed(self._get('d') + self._join('', self._get('ds')))])
    self._pop('dec_int_lit__c1')