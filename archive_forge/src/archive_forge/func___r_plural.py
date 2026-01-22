from .basestemmer import BaseStemmer
from .among import Among
def __r_plural(self):
    self.ket = self.cursor
    among_var = self.find_among_b(HungarianStemmer.a_8)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if not self.__r_R1():
        return False
    if among_var == 1:
        if not self.slice_from(u'a'):
            return False
    elif among_var == 2:
        if not self.slice_from(u'e'):
            return False
    elif not self.slice_del():
        return False
    return True