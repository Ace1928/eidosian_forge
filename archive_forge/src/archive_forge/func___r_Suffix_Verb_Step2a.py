from .basestemmer import BaseStemmer
from .among import Among
def __r_Suffix_Verb_Step2a(self):
    self.ket = self.cursor
    among_var = self.find_among_b(ArabicStemmer.a_18)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if among_var == 1:
        if not len(self.current) >= 4:
            return False
        if not self.slice_del():
            return False
    elif among_var == 2:
        if not len(self.current) >= 5:
            return False
        if not self.slice_del():
            return False
    elif among_var == 3:
        if not len(self.current) > 5:
            return False
        if not self.slice_del():
            return False
    else:
        if not len(self.current) >= 6:
            return False
        if not self.slice_del():
            return False
    return True