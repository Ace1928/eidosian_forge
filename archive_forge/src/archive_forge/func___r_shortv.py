from .basestemmer import BaseStemmer
from .among import Among
def __r_shortv(self):
    try:
        v_1 = self.limit - self.cursor
        try:
            if not self.out_grouping_b(EnglishStemmer.g_v_WXY, 89, 121):
                raise lab1()
            if not self.in_grouping_b(EnglishStemmer.g_v, 97, 121):
                raise lab1()
            if not self.out_grouping_b(EnglishStemmer.g_v, 97, 121):
                raise lab1()
            raise lab0()
        except lab1:
            pass
        self.cursor = self.limit - v_1
        if not self.out_grouping_b(EnglishStemmer.g_v, 97, 121):
            return False
        if not self.in_grouping_b(EnglishStemmer.g_v, 97, 121):
            return False
        if self.cursor > self.limit_backward:
            return False
    except lab0:
        pass
    return True