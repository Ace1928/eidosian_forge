import unicodedata
def _ws__c8__s0_n_n_g__c0__s1_(self):
    v = self._is_unicat(self._get('x'), 'Zs')
    if v:
        self._succeed(v)
    else:
        self._fail()