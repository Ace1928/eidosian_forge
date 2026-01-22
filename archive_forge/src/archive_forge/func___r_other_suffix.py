from .basestemmer import BaseStemmer
from .among import Among
def __r_other_suffix(self):
    if self.cursor < self.I_p1:
        return False
    v_2 = self.limit_backward
    self.limit_backward = self.I_p1
    self.ket = self.cursor
    if self.find_among_b(NorwegianStemmer.a_2) == 0:
        self.limit_backward = v_2
        return False
    self.bra = self.cursor
    self.limit_backward = v_2
    if not self.slice_del():
        return False
    return True