from .basestemmer import BaseStemmer
from .among import Among
def __r_other_endings(self):
    if self.cursor < self.I_p2:
        return False
    v_2 = self.limit_backward
    self.limit_backward = self.I_p2
    self.ket = self.cursor
    among_var = self.find_among_b(FinnishStemmer.a_7)
    if among_var == 0:
        self.limit_backward = v_2
        return False
    self.bra = self.cursor
    self.limit_backward = v_2
    if among_var == 1:
        v_3 = self.limit - self.cursor
        try:
            if not self.eq_s_b(u'po'):
                raise lab0()
            return False
        except lab0:
            pass
        self.cursor = self.limit - v_3
    if not self.slice_del():
        return False
    return True