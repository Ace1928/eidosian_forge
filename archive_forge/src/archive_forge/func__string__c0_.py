import unicodedata
def _string__c0_(self):
    self._push('string__c0')
    self._seq([self._squote_, self._string__c0__s1_, self._squote_, lambda: self._succeed(self._join('', self._get('cs')))])
    self._pop('string__c0')