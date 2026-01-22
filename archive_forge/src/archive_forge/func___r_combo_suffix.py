from .basestemmer import BaseStemmer
from .among import Among
def __r_combo_suffix(self):
    v_1 = self.limit - self.cursor
    self.ket = self.cursor
    among_var = self.find_among_b(RomanianStemmer.a_2)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if not self.__r_R1():
        return False
    if among_var == 1:
        if not self.slice_from(u'abil'):
            return False
    elif among_var == 2:
        if not self.slice_from(u'ibil'):
            return False
    elif among_var == 3:
        if not self.slice_from(u'iv'):
            return False
    elif among_var == 4:
        if not self.slice_from(u'ic'):
            return False
    elif among_var == 5:
        if not self.slice_from(u'at'):
            return False
    elif not self.slice_from(u'it'):
        return False
    self.B_standard_suffix_removed = True
    self.cursor = self.limit - v_1
    return True