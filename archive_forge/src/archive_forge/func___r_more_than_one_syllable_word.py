from .basestemmer import BaseStemmer
from .among import Among
def __r_more_than_one_syllable_word(self):
    v_1 = self.cursor
    v_2 = 2
    while True:
        v_3 = self.cursor
        try:
            if not self.go_out_grouping(TurkishStemmer.g_vowel, 97, 305):
                raise lab0()
            self.cursor += 1
            v_2 -= 1
            continue
        except lab0:
            pass
        self.cursor = v_3
        break
    if v_2 > 0:
        return False
    self.cursor = v_1
    return True