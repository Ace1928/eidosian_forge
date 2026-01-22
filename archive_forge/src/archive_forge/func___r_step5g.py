from .basestemmer import BaseStemmer
from .among import Among
def __r_step5g(self):
    v_1 = self.limit - self.cursor
    try:
        self.ket = self.cursor
        if self.find_among_b(GreekStemmer.a_47) == 0:
            raise lab0()
        self.bra = self.cursor
        if not self.slice_del():
            return False
        self.B_test1 = False
    except lab0:
        pass
    self.cursor = self.limit - v_1
    self.ket = self.cursor
    if self.find_among_b(GreekStemmer.a_50) == 0:
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
            if self.find_among_b(GreekStemmer.a_48) == 0:
                raise lab2()
            if not self.slice_from(u'ηκ'):
                return False
            raise lab1()
        except lab2:
            pass
        self.cursor = self.limit - v_2
        self.ket = self.cursor
        self.bra = self.cursor
        if self.find_among_b(GreekStemmer.a_49) == 0:
            return False
        if self.cursor > self.limit_backward:
            return False
        if not self.slice_from(u'ηκ'):
            return False
    except lab1:
        pass
    return True