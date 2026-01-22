import unicodedata
def _member__c1_(self):
    self._push('member__c1')
    self._seq([lambda: self._bind(self._ident_, 'k'), self._sp_, lambda: self._ch(':'), self._sp_, lambda: self._bind(self._value_, 'v'), lambda: self._succeed([self._get('k'), self._get('v')])])
    self._pop('member__c1')