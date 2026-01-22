from .basestemmer import BaseStemmer
from .among import Among
def __r_step5i(self):
    self.ket = self.cursor
    if self.find_among_b(GreekStemmer.a_56) == 0:
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
            if not self.eq_s_b(u'κολλ'):
                raise lab1()
            if not self.slice_from(u'αγ'):
                return False
            raise lab0()
        except lab1:
            pass
        self.cursor = self.limit - v_1
        try:
            v_2 = self.limit - self.cursor
            try:
                self.ket = self.cursor
                self.bra = self.cursor
                among_var = self.find_among_b(GreekStemmer.a_54)
                if among_var == 0:
                    raise lab3()
                if among_var == 1:
                    if not self.slice_from(u'αγ'):
                        return False
                raise lab2()
            except lab3:
                pass
            self.cursor = self.limit - v_2
            self.ket = self.cursor
            self.bra = self.cursor
            if self.find_among_b(GreekStemmer.a_55) == 0:
                return False
            if self.cursor > self.limit_backward:
                return False
            if not self.slice_from(u'αγ'):
                return False
        except lab2:
            pass
    except lab0:
        pass
    return True