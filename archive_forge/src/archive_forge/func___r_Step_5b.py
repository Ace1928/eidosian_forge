from .basestemmer import BaseStemmer
from .among import Among
def __r_Step_5b(self):
    self.ket = self.cursor
    if not self.eq_s_b(u'l'):
        return False
    self.bra = self.cursor
    if not self.__r_R2():
        return False
    if not self.eq_s_b(u'l'):
        return False
    if not self.slice_del():
        return False
    return True