from .basestemmer import BaseStemmer
from .among import Among
def __r_v_ending(self):
    self.ket = self.cursor
    among_var = self.find_among_b(HungarianStemmer.a_1)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if not self.__r_R1():
        return False
    if among_var == 1:
        if not self.slice_from(u'a'):
            return False
    elif not self.slice_from(u'e'):
        return False
    return True