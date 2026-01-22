from .basestemmer import BaseStemmer
from .among import Among
def __r_Prefix_Step1(self):
    self.bra = self.cursor
    among_var = self.find_among(ArabicStemmer.a_4)
    if among_var == 0:
        return False
    self.ket = self.cursor
    if among_var == 1:
        if not len(self.current) > 3:
            return False
        if not self.slice_from(u'أ'):
            return False
    elif among_var == 2:
        if not len(self.current) > 3:
            return False
        if not self.slice_from(u'آ'):
            return False
    elif among_var == 3:
        if not len(self.current) > 3:
            return False
        if not self.slice_from(u'ا'):
            return False
    else:
        if not len(self.current) > 3:
            return False
        if not self.slice_from(u'إ'):
            return False
    return True