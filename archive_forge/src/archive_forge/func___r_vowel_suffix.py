from .basestemmer import BaseStemmer
from .among import Among
def __r_vowel_suffix(self):
    v_1 = self.limit - self.cursor
    try:
        self.ket = self.cursor
        if not self.in_grouping_b(ItalianStemmer.g_AEIO, 97, 242):
            self.cursor = self.limit - v_1
            raise lab0()
        self.bra = self.cursor
        if not self.__r_RV():
            self.cursor = self.limit - v_1
            raise lab0()
        if not self.slice_del():
            return False
        self.ket = self.cursor
        if not self.eq_s_b(u'i'):
            self.cursor = self.limit - v_1
            raise lab0()
        self.bra = self.cursor
        if not self.__r_RV():
            self.cursor = self.limit - v_1
            raise lab0()
        if not self.slice_del():
            return False
    except lab0:
        pass
    v_2 = self.limit - self.cursor
    try:
        self.ket = self.cursor
        if not self.eq_s_b(u'h'):
            self.cursor = self.limit - v_2
            raise lab1()
        self.bra = self.cursor
        if not self.in_grouping_b(ItalianStemmer.g_CG, 99, 103):
            self.cursor = self.limit - v_2
            raise lab1()
        if not self.__r_RV():
            self.cursor = self.limit - v_2
            raise lab1()
        if not self.slice_del():
            return False
    except lab1:
        pass
    return True