from .basestemmer import BaseStemmer
from .among import Among
def __r_mark_suffix_with_optional_n_consonant(self):
    try:
        v_1 = self.limit - self.cursor
        try:
            if not self.eq_s_b(u'n'):
                raise lab1()
            v_2 = self.limit - self.cursor
            if not self.in_grouping_b(TurkishStemmer.g_vowel, 97, 305):
                raise lab1()
            self.cursor = self.limit - v_2
            raise lab0()
        except lab1:
            pass
        self.cursor = self.limit - v_1
        v_3 = self.limit - self.cursor
        try:
            v_4 = self.limit - self.cursor
            if not self.eq_s_b(u'n'):
                raise lab2()
            self.cursor = self.limit - v_4
            return False
        except lab2:
            pass
        self.cursor = self.limit - v_3
        v_5 = self.limit - self.cursor
        if self.cursor <= self.limit_backward:
            return False
        self.cursor -= 1
        if not self.in_grouping_b(TurkishStemmer.g_vowel, 97, 305):
            return False
        self.cursor = self.limit - v_5
    except lab0:
        pass
    return True