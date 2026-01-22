from .basestemmer import BaseStemmer
from .among import Among
def __r_step5f(self):
    v_1 = self.limit - self.cursor
    try:
        self.ket = self.cursor
        if not self.eq_s_b(u'ιεστε'):
            raise lab0()
        self.bra = self.cursor
        if not self.slice_del():
            return False
        self.B_test1 = False
        self.ket = self.cursor
        self.bra = self.cursor
        if self.find_among_b(GreekStemmer.a_45) == 0:
            raise lab0()
        if self.cursor > self.limit_backward:
            raise lab0()
        if not self.slice_from(u'ιεστ'):
            return False
    except lab0:
        pass
    self.cursor = self.limit - v_1
    self.ket = self.cursor
    if not self.eq_s_b(u'εστε'):
        return False
    self.bra = self.cursor
    if not self.slice_del():
        return False
    self.B_test1 = False
    self.ket = self.cursor
    self.bra = self.cursor
    if self.find_among_b(GreekStemmer.a_46) == 0:
        return False
    if self.cursor > self.limit_backward:
        return False
    if not self.slice_from(u'ιεστ'):
        return False
    return True