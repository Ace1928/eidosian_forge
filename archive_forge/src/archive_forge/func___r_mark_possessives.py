from .basestemmer import BaseStemmer
from .among import Among
def __r_mark_possessives(self):
    if self.find_among_b(TurkishStemmer.a_0) == 0:
        return False
    if not self.__r_mark_suffix_with_optional_U_vowel():
        return False
    return True