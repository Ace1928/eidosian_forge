from .basestemmer import BaseStemmer
from .among import Among
def __r_steps7(self):
    self.ket = self.cursor
    if self.find_among_b(GreekStemmer.a_16) == 0:
        return False
    self.bra = self.cursor
    if not self.slice_del():
        return False
    self.B_test1 = False
    self.ket = self.cursor
    self.bra = self.cursor
    if self.find_among_b(GreekStemmer.a_15) == 0:
        return False
    if self.cursor > self.limit_backward:
        return False
    if not self.slice_from(u'αρακ'):
        return False
    return True