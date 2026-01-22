import unicodedata
def _value__c3_(self):
    self._push('value__c3')
    self._seq([lambda: self._bind(self._object_, 'v'), lambda: self._succeed(['object', self._get('v')])])
    self._pop('value__c3')