from .basestemmer import BaseStemmer
from .among import Among
def __r_i_verb_suffix(self):
    if self.cursor < self.I_pV:
        return False
    v_2 = self.limit_backward
    self.limit_backward = self.I_pV
    self.ket = self.cursor
    if self.find_among_b(FrenchStemmer.a_5) == 0:
        self.limit_backward = v_2
        return False
    self.bra = self.cursor
    v_3 = self.limit - self.cursor
    try:
        if not self.eq_s_b(u'H'):
            raise lab0()
        self.limit_backward = v_2
        return False
    except lab0:
        pass
    self.cursor = self.limit - v_3
    if not self.out_grouping_b(FrenchStemmer.g_v, 97, 251):
        self.limit_backward = v_2
        return False
    if not self.slice_del():
        return False
    self.limit_backward = v_2
    return True