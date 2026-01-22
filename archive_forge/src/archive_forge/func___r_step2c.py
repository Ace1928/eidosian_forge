from .basestemmer import BaseStemmer
from .among import Among
def __r_step2c(self):
    self.ket = self.cursor
    if self.find_among_b(GreekStemmer.a_28) == 0:
        return False
    self.bra = self.cursor
    if not self.slice_del():
        return False
    self.ket = self.cursor
    self.bra = self.cursor
    if self.find_among_b(GreekStemmer.a_29) == 0:
        return False
    if not self.slice_from(u'ουδ'):
        return False
    return True