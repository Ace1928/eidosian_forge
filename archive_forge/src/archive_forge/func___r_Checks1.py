from .basestemmer import BaseStemmer
from .among import Among
def __r_Checks1(self):
    self.bra = self.cursor
    among_var = self.find_among(ArabicStemmer.a_3)
    if among_var == 0:
        return False
    self.ket = self.cursor
    if among_var == 1:
        if not len(self.current) > 4:
            return False
        self.B_is_noun = True
        self.B_is_verb = False
        self.B_is_defined = True
    else:
        if not len(self.current) > 3:
            return False
        self.B_is_noun = True
        self.B_is_verb = False
        self.B_is_defined = True
    return True