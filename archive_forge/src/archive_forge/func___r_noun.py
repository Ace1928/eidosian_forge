from .basestemmer import BaseStemmer
from .among import Among
def __r_noun(self):
    self.ket = self.cursor
    if self.find_among_b(ArmenianStemmer.a_2) == 0:
        return False
    self.bra = self.cursor
    if not self.slice_del():
        return False
    return True