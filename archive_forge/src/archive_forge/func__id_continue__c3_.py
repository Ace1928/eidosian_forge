import unicodedata
def _id_continue__c3_(self):
    self._push('id_continue__c3')
    self._seq([lambda: self._bind(self._anything_, 'x'), self._id_continue__c3__s1_, lambda: self._succeed(self._get('x'))])
    self._pop('id_continue__c3')