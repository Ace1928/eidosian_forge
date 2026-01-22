from .basestemmer import BaseStemmer
from .among import Among
def __r_steps3(self):
    self.ket = self.cursor
    if self.find_among_b(GreekStemmer.a_7) == 0:
        return False
    self.bra = self.cursor
    if not self.slice_del():
        return False
    self.B_test1 = False
    try:
        v_1 = self.limit - self.cursor
        try:
            if not self.eq_s_b(u'ισα'):
                raise lab1()
            if self.cursor > self.limit_backward:
                raise lab1()
            if not self.slice_from(u'ισ'):
                return False
            raise lab0()
        except lab1:
            pass
        self.cursor = self.limit - v_1
        self.ket = self.cursor
        self.bra = self.cursor
        among_var = self.find_among_b(GreekStemmer.a_6)
        if among_var == 0:
            return False
        if self.cursor > self.limit_backward:
            return False
        if among_var == 1:
            if not self.slice_from(u'ι'):
                return False
        elif not self.slice_from(u'ισ'):
            return False
    except lab0:
        pass
    return True