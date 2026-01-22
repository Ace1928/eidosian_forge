from .basestemmer import BaseStemmer
from .among import Among
def __r_y_verb_suffix(self):
    if self.cursor < self.I_pV:
        return False
    v_2 = self.limit_backward
    self.limit_backward = self.I_pV
    self.ket = self.cursor
    if self.find_among_b(SpanishStemmer.a_7) == 0:
        self.limit_backward = v_2
        return False
    self.bra = self.cursor
    self.limit_backward = v_2
    if not self.eq_s_b(u'u'):
        return False
    if not self.slice_del():
        return False
    return True