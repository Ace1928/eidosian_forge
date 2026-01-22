from .basestemmer import BaseStemmer
from .among import Among
def __r_Suffix_All_alef_maqsura(self):
    self.ket = self.cursor
    if self.find_among_b(ArabicStemmer.a_21) == 0:
        return False
    self.bra = self.cursor
    if not self.slice_from(u'ي'):
        return False
    return True