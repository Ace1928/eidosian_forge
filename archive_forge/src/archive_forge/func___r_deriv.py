from .basestemmer import BaseStemmer
from .among import Among
def __r_deriv(self):
    self.ket = self.cursor
    among_var = self.find_among_b(IrishStemmer.a_2)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if among_var == 1:
        if not self.__r_R2():
            return False
        if not self.slice_del():
            return False
    elif among_var == 2:
        if not self.slice_from(u'arc'):
            return False
    elif among_var == 3:
        if not self.slice_from(u'gin'):
            return False
    elif among_var == 4:
        if not self.slice_from(u'graf'):
            return False
    elif among_var == 5:
        if not self.slice_from(u'paite'):
            return False
    elif not self.slice_from(u'óid'):
        return False
    return True