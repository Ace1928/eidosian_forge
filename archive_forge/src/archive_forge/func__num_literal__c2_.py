import unicodedata
def _num_literal__c2_(self):
    self._push('num_literal__c2')
    self._seq([lambda: self._bind(self._dec_literal_, 'd'), lambda: self._not(self._id_start_), lambda: self._succeed(self._get('d'))])
    self._pop('num_literal__c2')