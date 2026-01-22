from .basestemmer import BaseStemmer
from .among import Among
def __r_remove_first_order_prefix(self):
    self.bra = self.cursor
    among_var = self.find_among(IndonesianStemmer.a_3)
    if among_var == 0:
        return False
    self.ket = self.cursor
    if among_var == 1:
        if not self.slice_del():
            return False
        self.I_prefix = 1
        self.I_measure -= 1
    elif among_var == 2:
        if not self.slice_del():
            return False
        self.I_prefix = 3
        self.I_measure -= 1
    elif among_var == 3:
        self.I_prefix = 1
        if not self.slice_from(u's'):
            return False
        self.I_measure -= 1
    elif among_var == 4:
        self.I_prefix = 3
        if not self.slice_from(u's'):
            return False
        self.I_measure -= 1
    elif among_var == 5:
        self.I_prefix = 1
        self.I_measure -= 1
        try:
            v_1 = self.cursor
            try:
                v_2 = self.cursor
                if not self.in_grouping(IndonesianStemmer.g_vowel, 97, 117):
                    raise lab1()
                self.cursor = v_2
                if not self.slice_from(u'p'):
                    return False
                raise lab0()
            except lab1:
                pass
            self.cursor = v_1
            if not self.slice_del():
                return False
        except lab0:
            pass
    else:
        self.I_prefix = 3
        self.I_measure -= 1
        try:
            v_3 = self.cursor
            try:
                v_4 = self.cursor
                if not self.in_grouping(IndonesianStemmer.g_vowel, 97, 117):
                    raise lab3()
                self.cursor = v_4
                if not self.slice_from(u'p'):
                    return False
                raise lab2()
            except lab3:
                pass
            self.cursor = v_3
            if not self.slice_del():
                return False
        except lab2:
            pass
    return True