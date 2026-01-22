from .basestemmer import BaseStemmer
from .among import Among
def __r_postlude(self):
    while True:
        v_1 = self.cursor
        try:
            self.bra = self.cursor
            among_var = self.find_among(ItalianStemmer.a_1)
            if among_var == 0:
                raise lab0()
            self.ket = self.cursor
            if among_var == 1:
                if not self.slice_from(u'i'):
                    return False
            elif among_var == 2:
                if not self.slice_from(u'u'):
                    return False
            else:
                if self.cursor >= self.limit:
                    raise lab0()
                self.cursor += 1
            continue
        except lab0:
            pass
        self.cursor = v_1
        break
    return True