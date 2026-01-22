import unicodedata
def _dqchar__c1_(self):
    self._seq([self._bslash_, self._eol_, lambda: self._succeed('')])