from .basestemmer import BaseStemmer
from .among import Among
def __r_Step_1c(self):
    self.ket = self.cursor
    try:
        v_1 = self.limit - self.cursor
        try:
            if not self.eq_s_b(u'y'):
                raise lab1()
            raise lab0()
        except lab1:
            pass
        self.cursor = self.limit - v_1
        if not self.eq_s_b(u'Y'):
            return False
    except lab0:
        pass
    self.bra = self.cursor
    if not self.out_grouping_b(EnglishStemmer.g_v, 97, 121):
        return False
    try:
        if self.cursor > self.limit_backward:
            raise lab2()
        return False
    except lab2:
        pass
    if not self.slice_from(u'i'):
        return False
    return True