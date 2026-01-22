import unicodedata
def _array__c0_(self):
    self._push('array__c0')
    self._seq([lambda: self._ch('['), self._sp_, lambda: self._bind(self._element_list_, 'v'), self._sp_, lambda: self._ch(']'), lambda: self._succeed(self._get('v'))])
    self._pop('array__c0')