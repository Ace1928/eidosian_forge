from .basestemmer import BaseStemmer
from .among import Among
def __r_perfective_gerund(self):
    self.ket = self.cursor
    among_var = self.find_among_b(RussianStemmer.a_0)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if among_var == 1:
        try:
            v_1 = self.limit - self.cursor
            try:
                if not self.eq_s_b(u'а'):
                    raise lab1()
                raise lab0()
            except lab1:
                pass
            self.cursor = self.limit - v_1
            if not self.eq_s_b(u'я'):
                return False
        except lab0:
            pass
        if not self.slice_del():
            return False
    elif not self.slice_del():
        return False
    return True