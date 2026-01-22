from .basestemmer import BaseStemmer
from .among import Among
def __r_main_suffix(self):
    if self.cursor < self.I_p1:
        return False
    v_2 = self.limit_backward
    self.limit_backward = self.I_p1
    self.ket = self.cursor
    among_var = self.find_among_b(NorwegianStemmer.a_0)
    if among_var == 0:
        self.limit_backward = v_2
        return False
    self.bra = self.cursor
    self.limit_backward = v_2
    if among_var == 1:
        if not self.slice_del():
            return False
    elif among_var == 2:
        try:
            v_3 = self.limit - self.cursor
            try:
                if not self.in_grouping_b(NorwegianStemmer.g_s_ending, 98, 122):
                    raise lab1()
                raise lab0()
            except lab1:
                pass
            self.cursor = self.limit - v_3
            if not self.eq_s_b(u'k'):
                return False
            if not self.out_grouping_b(NorwegianStemmer.g_v, 97, 248):
                return False
        except lab0:
            pass
        if not self.slice_del():
            return False
    elif not self.slice_from(u'er'):
        return False
    return True