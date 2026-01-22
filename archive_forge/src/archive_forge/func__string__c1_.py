import unicodedata
def _string__c1_(self):
    self._push('string__c1')
    self._seq([self._dquote_, self._string__c1__s1_, self._dquote_, lambda: self._succeed(self._join('', self._get('cs')))])
    self._pop('string__c1')