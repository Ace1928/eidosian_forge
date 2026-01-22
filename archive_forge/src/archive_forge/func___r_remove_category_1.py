from .basestemmer import BaseStemmer
from .among import Among
def __r_remove_category_1(self):
    self.ket = self.cursor
    among_var = self.find_among_b(NepaliStemmer.a_0)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if among_var == 1:
        if not self.slice_del():
            return False
    else:
        try:
            v_1 = self.limit - self.cursor
            try:
                try:
                    v_2 = self.limit - self.cursor
                    try:
                        if not self.eq_s_b(u'ए'):
                            raise lab3()
                        raise lab2()
                    except lab3:
                        pass
                    self.cursor = self.limit - v_2
                    if not self.eq_s_b(u'े'):
                        raise lab1()
                except lab2:
                    pass
                raise lab0()
            except lab1:
                pass
            self.cursor = self.limit - v_1
            if not self.slice_del():
                return False
        except lab0:
            pass
    return True