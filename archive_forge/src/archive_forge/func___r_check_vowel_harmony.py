from .basestemmer import BaseStemmer
from .among import Among
def __r_check_vowel_harmony(self):
    v_1 = self.limit - self.cursor
    if not self.go_out_grouping_b(TurkishStemmer.g_vowel, 97, 305):
        return False
    try:
        v_2 = self.limit - self.cursor
        try:
            if not self.eq_s_b(u'a'):
                raise lab1()
            if not self.go_out_grouping_b(TurkishStemmer.g_vowel1, 97, 305):
                raise lab1()
            raise lab0()
        except lab1:
            pass
        self.cursor = self.limit - v_2
        try:
            if not self.eq_s_b(u'e'):
                raise lab2()
            if not self.go_out_grouping_b(TurkishStemmer.g_vowel2, 101, 252):
                raise lab2()
            raise lab0()
        except lab2:
            pass
        self.cursor = self.limit - v_2
        try:
            if not self.eq_s_b(u'ı'):
                raise lab3()
            if not self.go_out_grouping_b(TurkishStemmer.g_vowel3, 97, 305):
                raise lab3()
            raise lab0()
        except lab3:
            pass
        self.cursor = self.limit - v_2
        try:
            if not self.eq_s_b(u'i'):
                raise lab4()
            if not self.go_out_grouping_b(TurkishStemmer.g_vowel4, 101, 105):
                raise lab4()
            raise lab0()
        except lab4:
            pass
        self.cursor = self.limit - v_2
        try:
            if not self.eq_s_b(u'o'):
                raise lab5()
            if not self.go_out_grouping_b(TurkishStemmer.g_vowel5, 111, 117):
                raise lab5()
            raise lab0()
        except lab5:
            pass
        self.cursor = self.limit - v_2
        try:
            if not self.eq_s_b(u'ö'):
                raise lab6()
            if not self.go_out_grouping_b(TurkishStemmer.g_vowel6, 246, 252):
                raise lab6()
            raise lab0()
        except lab6:
            pass
        self.cursor = self.limit - v_2
        try:
            if not self.eq_s_b(u'u'):
                raise lab7()
            if not self.go_out_grouping_b(TurkishStemmer.g_vowel5, 111, 117):
                raise lab7()
            raise lab0()
        except lab7:
            pass
        self.cursor = self.limit - v_2
        if not self.eq_s_b(u'ü'):
            return False
        if not self.go_out_grouping_b(TurkishStemmer.g_vowel6, 246, 252):
            return False
    except lab0:
        pass
    self.cursor = self.limit - v_1
    return True