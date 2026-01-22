from .basestemmer import BaseStemmer
from .among import Among
def __r_remove_plural_suffix(self):
    self.B_found_a_match = False
    self.limit_backward = self.cursor
    self.cursor = self.limit
    try:
        v_1 = self.limit - self.cursor
        try:
            self.ket = self.cursor
            if not self.eq_s_b(u'ுங்கள்'):
                raise lab1()
            v_2 = self.limit - self.cursor
            v_3 = self.limit - self.cursor
            try:
                if self.find_among_b(TamilStemmer.a_13) == 0:
                    raise lab2()
                raise lab1()
            except lab2:
                pass
            self.cursor = self.limit - v_3
            self.cursor = self.limit - v_2
            self.bra = self.cursor
            if not self.slice_from(u'்'):
                return False
            raise lab0()
        except lab1:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            if not self.eq_s_b(u'ற்கள்'):
                raise lab3()
            self.bra = self.cursor
            if not self.slice_from(u'ல்'):
                return False
            raise lab0()
        except lab3:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            if not self.eq_s_b(u'ட்கள்'):
                raise lab4()
            self.bra = self.cursor
            if not self.slice_from(u'ள்'):
                return False
            raise lab0()
        except lab4:
            pass
        self.cursor = self.limit - v_1
        self.ket = self.cursor
        if not self.eq_s_b(u'கள்'):
            return False
        self.bra = self.cursor
        if not self.slice_del():
            return False
    except lab0:
        pass
    self.B_found_a_match = True
    self.cursor = self.limit_backward
    return True