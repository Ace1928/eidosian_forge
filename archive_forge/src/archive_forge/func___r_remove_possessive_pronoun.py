from .basestemmer import BaseStemmer
from .among import Among
def __r_remove_possessive_pronoun(self):
    self.ket = self.cursor
    if self.find_among_b(IndonesianStemmer.a_1) == 0:
        return False
    self.bra = self.cursor
    if not self.slice_del():
        return False
    self.I_measure -= 1
    return True