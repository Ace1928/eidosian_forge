from .basestemmer import BaseStemmer
from .among import Among
def __r_fix_chdz(self):
    self.ket = self.cursor
    among_var = self.find_among_b(LithuanianStemmer.a_3)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if among_var == 1:
        if not self.slice_from(u't'):
            return False
    elif not self.slice_from(u'd'):
        return False
    return True