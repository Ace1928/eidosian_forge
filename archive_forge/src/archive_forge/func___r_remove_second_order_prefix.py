from .basestemmer import BaseStemmer
from .among import Among
def __r_remove_second_order_prefix(self):
    self.bra = self.cursor
    among_var = self.find_among(IndonesianStemmer.a_4)
    if among_var == 0:
        return False
    self.ket = self.cursor
    if among_var == 1:
        if not self.slice_del():
            return False
        self.I_prefix = 2
        self.I_measure -= 1
    elif among_var == 2:
        if not self.slice_from(u'ajar'):
            return False
        self.I_measure -= 1
    elif among_var == 3:
        if not self.slice_del():
            return False
        self.I_prefix = 4
        self.I_measure -= 1
    else:
        if not self.slice_from(u'ajar'):
            return False
        self.I_prefix = 4
        self.I_measure -= 1
    return True