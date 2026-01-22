from .basestemmer import BaseStemmer
from .among import Among
def __r_tidy(self):
    if self.cursor < self.I_p1:
        return False
    v_2 = self.limit_backward
    self.limit_backward = self.I_p1
    v_3 = self.limit - self.cursor
    try:
        v_4 = self.limit - self.cursor
        if not self.__r_LONG():
            raise lab0()
        self.cursor = self.limit - v_4
        self.ket = self.cursor
        if self.cursor <= self.limit_backward:
            raise lab0()
        self.cursor -= 1
        self.bra = self.cursor
        if not self.slice_del():
            return False
    except lab0:
        pass
    self.cursor = self.limit - v_3
    v_5 = self.limit - self.cursor
    try:
        self.ket = self.cursor
        if not self.in_grouping_b(FinnishStemmer.g_AEI, 97, 228):
            raise lab1()
        self.bra = self.cursor
        if not self.in_grouping_b(FinnishStemmer.g_C, 98, 122):
            raise lab1()
        if not self.slice_del():
            return False
    except lab1:
        pass
    self.cursor = self.limit - v_5
    v_6 = self.limit - self.cursor
    try:
        self.ket = self.cursor
        if not self.eq_s_b(u'j'):
            raise lab2()
        self.bra = self.cursor
        try:
            v_7 = self.limit - self.cursor
            try:
                if not self.eq_s_b(u'o'):
                    raise lab4()
                raise lab3()
            except lab4:
                pass
            self.cursor = self.limit - v_7
            if not self.eq_s_b(u'u'):
                raise lab2()
        except lab3:
            pass
        if not self.slice_del():
            return False
    except lab2:
        pass
    self.cursor = self.limit - v_6
    v_8 = self.limit - self.cursor
    try:
        self.ket = self.cursor
        if not self.eq_s_b(u'o'):
            raise lab5()
        self.bra = self.cursor
        if not self.eq_s_b(u'j'):
            raise lab5()
        if not self.slice_del():
            return False
    except lab5:
        pass
    self.cursor = self.limit - v_8
    self.limit_backward = v_2
    if not self.go_in_grouping_b(FinnishStemmer.g_V1, 97, 246):
        return False
    self.ket = self.cursor
    if not self.in_grouping_b(FinnishStemmer.g_C, 98, 122):
        return False
    self.bra = self.cursor
    self.S_x = self.slice_to()
    if self.S_x == '':
        return False
    if not self.eq_s_b(self.S_x):
        return False
    if not self.slice_del():
        return False
    return True