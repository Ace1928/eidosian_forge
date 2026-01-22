from .basestemmer import BaseStemmer
from .among import Among
def __r_exception1(self):
    self.bra = self.cursor
    among_var = self.find_among(EnglishStemmer.a_10)
    if among_var == 0:
        return False
    self.ket = self.cursor
    if self.cursor < self.limit:
        return False
    if among_var == 1:
        if not self.slice_from(u'ski'):
            return False
    elif among_var == 2:
        if not self.slice_from(u'sky'):
            return False
    elif among_var == 3:
        if not self.slice_from(u'die'):
            return False
    elif among_var == 4:
        if not self.slice_from(u'lie'):
            return False
    elif among_var == 5:
        if not self.slice_from(u'tie'):
            return False
    elif among_var == 6:
        if not self.slice_from(u'idl'):
            return False
    elif among_var == 7:
        if not self.slice_from(u'gentl'):
            return False
    elif among_var == 8:
        if not self.slice_from(u'ugli'):
            return False
    elif among_var == 9:
        if not self.slice_from(u'earli'):
            return False
    elif among_var == 10:
        if not self.slice_from(u'onli'):
            return False
    elif among_var == 11:
        if not self.slice_from(u'singl'):
            return False
    return True