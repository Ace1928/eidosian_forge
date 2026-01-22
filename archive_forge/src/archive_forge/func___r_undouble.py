from .basestemmer import BaseStemmer
from .among import Among
def __r_undouble(self):
    if self.cursor < self.I_p1:
        return False
    v_2 = self.limit_backward
    self.limit_backward = self.I_p1
    self.ket = self.cursor
    if not self.in_grouping_b(DanishStemmer.g_c, 98, 122):
        self.limit_backward = v_2
        return False
    self.bra = self.cursor
    self.S_ch = self.slice_to()
    if self.S_ch == '':
        return False
    self.limit_backward = v_2
    if not self.eq_s_b(self.S_ch):
        return False
    if not self.slice_del():
        return False
    return True