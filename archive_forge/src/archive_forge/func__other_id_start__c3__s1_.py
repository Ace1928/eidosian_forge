import unicodedata
def _other_id_start__c3__s1_(self):
    v = self._is_unicat(self._get('x'), 'Lt')
    if v:
        self._succeed(v)
    else:
        self._fail()