import unicodedata
def _dec_literal__c3_(self):
    self._push('dec_literal__c3')
    self._seq([lambda: self._bind(self._dec_int_lit_, 'd'), lambda: self._succeed(self._get('d'))])
    self._pop('dec_literal__c3')