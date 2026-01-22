import unicodedata
def _ws__c8_(self):
    self._push('ws__c8')
    self._seq([self._ws__c8__s0_, lambda: self._bind(self._anything_, 'x'), lambda: self._succeed(self._get('x'))])
    self._pop('ws__c8')