import unicodedata
def _id_continue__c3__s1_(self):
    v = self._is_unicat(self._get('x'), 'Mn')
    if v:
        self._succeed(v)
    else:
        self._fail()