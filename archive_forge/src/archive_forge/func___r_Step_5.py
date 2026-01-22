from .basestemmer import BaseStemmer
from .among import Among
def __r_Step_5(self):
    self.ket = self.cursor
    among_var = self.find_among_b(EnglishStemmer.a_8)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if among_var == 1:
        try:
            v_1 = self.limit - self.cursor
            try:
                if not self.__r_R2():
                    raise lab1()
                raise lab0()
            except lab1:
                pass
            self.cursor = self.limit - v_1
            if not self.__r_R1():
                return False
            v_2 = self.limit - self.cursor
            try:
                if not self.__r_shortv():
                    raise lab2()
                return False
            except lab2:
                pass
            self.cursor = self.limit - v_2
        except lab0:
            pass
        if not self.slice_del():
            return False
    else:
        if not self.__r_R2():
            return False
        if not self.eq_s_b(u'l'):
            return False
        if not self.slice_del():
            return False
    return True