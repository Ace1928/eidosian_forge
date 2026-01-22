from .basestemmer import BaseStemmer
from .among import Among
def __r_noun_sfx(self):
    self.ket = self.cursor
    among_var = self.find_among_b(IrishStemmer.a_1)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if among_var == 1:
        if not self.__r_R1():
            return False
        if not self.slice_del():
            return False
    else:
        if not self.__r_R2():
            return False
        if not self.slice_del():
            return False
    return True