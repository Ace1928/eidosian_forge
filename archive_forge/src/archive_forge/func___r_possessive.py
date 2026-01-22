from .basestemmer import BaseStemmer
from .among import Among
def __r_possessive(self):
    if self.cursor < self.I_p1:
        return False
    v_2 = self.limit_backward
    self.limit_backward = self.I_p1
    self.ket = self.cursor
    among_var = self.find_among_b(FinnishStemmer.a_4)
    if among_var == 0:
        self.limit_backward = v_2
        return False
    self.bra = self.cursor
    self.limit_backward = v_2
    if among_var == 1:
        v_3 = self.limit - self.cursor
        try:
            if not self.eq_s_b(u'k'):
                raise lab0()
            return False
        except lab0:
            pass
        self.cursor = self.limit - v_3
        if not self.slice_del():
            return False
    elif among_var == 2:
        if not self.slice_del():
            return False
        self.ket = self.cursor
        if not self.eq_s_b(u'kse'):
            return False
        self.bra = self.cursor
        if not self.slice_from(u'ksi'):
            return False
    elif among_var == 3:
        if not self.slice_del():
            return False
    elif among_var == 4:
        if self.find_among_b(FinnishStemmer.a_1) == 0:
            return False
        if not self.slice_del():
            return False
    elif among_var == 5:
        if self.find_among_b(FinnishStemmer.a_2) == 0:
            return False
        if not self.slice_del():
            return False
    else:
        if self.find_among_b(FinnishStemmer.a_3) == 0:
            return False
        if not self.slice_del():
            return False
    return True