import unicodedata
def _value__c6_(self):
    self._push('value__c6')
    self._seq([lambda: self._bind(self._num_literal_, 'v'), lambda: self._succeed(['number', self._get('v')])])
    self._pop('value__c6')