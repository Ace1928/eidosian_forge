from .basestemmer import BaseStemmer
from .among import Among
def __r_step4(self):
    self.ket = self.cursor
    if self.find_among_b(GreekStemmer.a_33) == 0:
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
            if not self.in_grouping_b(GreekStemmer.g_v, 945, 969):
                raise lab1()
            if not self.slice_from(u'ικ'):
                return False
            raise lab0()
        except lab1:
            pass
        self.cursor = self.limit - v_1
        self.ket = self.cursor
    except lab0:
        pass
    self.bra = self.cursor
    if self.find_among_b(GreekStemmer.a_34) == 0:
        return False
    if self.cursor > self.limit_backward:
        return False
    if not self.slice_from(u'ικ'):
        return False
    return True