from .basestemmer import BaseStemmer
from .among import Among
def __r_remove_question_prefixes(self):
    self.bra = self.cursor
    if not self.eq_s(u'எ'):
        return False
    if self.find_among(TamilStemmer.a_0) == 0:
        return False
    if not self.eq_s(u'்'):
        return False
    self.ket = self.cursor
    if not self.slice_del():
        return False
    v_1 = self.cursor
    self.__r_fix_va_start()
    self.cursor = v_1
    return True