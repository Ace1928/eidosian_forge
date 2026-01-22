from .basestemmer import BaseStemmer
from .among import Among
def __r_Step_3(self):
    self.ket = self.cursor
    among_var = self.find_among_b(EnglishStemmer.a_6)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if not self.__r_R1():
        return False
    if among_var == 1:
        if not self.slice_from(u'tion'):
            return False
    elif among_var == 2:
        if not self.slice_from(u'ate'):
            return False
    elif among_var == 3:
        if not self.slice_from(u'al'):
            return False
    elif among_var == 4:
        if not self.slice_from(u'ic'):
            return False
    elif among_var == 5:
        if not self.slice_del():
            return False
    else:
        if not self.__r_R2():
            return False
        if not self.slice_del():
            return False
    return True