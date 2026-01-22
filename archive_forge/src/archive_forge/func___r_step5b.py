from .basestemmer import BaseStemmer
from .among import Among
def __r_step5b(self):
    v_1 = self.limit - self.cursor
    try:
        self.ket = self.cursor
        if self.find_among_b(GreekStemmer.a_38) == 0:
            raise lab0()
        self.bra = self.cursor
        if not self.slice_del():
            return False
        self.B_test1 = False
        self.ket = self.cursor
        self.bra = self.cursor
        if self.find_among_b(GreekStemmer.a_37) == 0:
            raise lab0()
        if self.cursor > self.limit_backward:
            raise lab0()
        if not self.slice_from(u'αγαν'):
            return False
    except lab0:
        pass
    self.cursor = self.limit - v_1
    self.ket = self.cursor
    if not self.eq_s_b(u'ανε'):
        return False
    self.bra = self.cursor
    if not self.slice_del():
        return False
    self.B_test1 = False
    try:
        v_2 = self.limit - self.cursor
        try:
            self.ket = self.cursor
            self.bra = self.cursor
            if not self.in_grouping_b(GreekStemmer.g_v2, 945, 969):
                raise lab2()
            if not self.slice_from(u'αν'):
                return False
            raise lab1()
        except lab2:
            pass
        self.cursor = self.limit - v_2
        self.ket = self.cursor
    except lab1:
        pass
    self.bra = self.cursor
    if self.find_among_b(GreekStemmer.a_39) == 0:
        return False
    if self.cursor > self.limit_backward:
        return False
    if not self.slice_from(u'αν'):
        return False
    return True