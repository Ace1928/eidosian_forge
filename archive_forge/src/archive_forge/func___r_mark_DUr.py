from .basestemmer import BaseStemmer
from .among import Among
def __r_mark_DUr(self):
    if not self.__r_check_vowel_harmony():
        return False
    if self.find_among_b(TurkishStemmer.a_18) == 0:
        return False
    return True