from .basestemmer import BaseStemmer
from .among import Among
def __r_attached_pronoun(self):
    self.ket = self.cursor
    if self.find_among_b(CatalanStemmer.a_1) == 0:
        return False
    self.bra = self.cursor
    if not self.__r_R1():
        return False
    if not self.slice_del():
        return False
    return True