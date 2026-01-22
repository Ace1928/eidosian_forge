import unicodedata
def _other_id_start__c4__s1_(self):
    v = self._is_unicat(self._get('x'), 'Lu')
    if v:
        self._succeed(v)
    else:
        self._fail()