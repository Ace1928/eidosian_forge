from .basestemmer import BaseStemmer
from .among import Among
def __r_step5d(self):
    self.ket = self.cursor
    if self.find_among_b(GreekStemmer.a_43) == 0:
        return False
    self.bra = self.cursor
    if not self.slice_del():
        return False
    self.B_test1 = False
    try:
        v_1 = self.limit - self.cursor
        try:
            self.ket = self.cursor
            self.bra = self.cursor
            if not self.eq_s_b(u'αρχ'):
                raise lab1()
            if self.cursor > self.limit_backward:
                raise lab1()
            if not self.slice_from(u'οντ'):
                return False
            raise lab0()
        except lab1:
            pass
        self.cursor = self.limit - v_1
        self.ket = self.cursor
        self.bra = self.cursor
        if not self.eq_s_b(u'κρε'):
            return False
        if not self.slice_from(u'ωντ'):
            return False
    except lab0:
        pass
    return True