from .basestemmer import BaseStemmer
from .among import Among
def __r_ending(self):
    self.ket = self.cursor
    if self.find_among_b(ArmenianStemmer.a_3) == 0:
        return False
    self.bra = self.cursor
    if not self.__r_R2():
        return False
    if not self.slice_del():
        return False
    return True