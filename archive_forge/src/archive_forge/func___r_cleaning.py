from .basestemmer import BaseStemmer
from .among import Among
def __r_cleaning(self):
    while True:
        v_1 = self.cursor
        try:
            self.bra = self.cursor
            among_var = self.find_among(CatalanStemmer.a_0)
            if among_var == 0:
                raise lab0()
            self.ket = self.cursor
            if among_var == 1:
                if not self.slice_from(u'a'):
                    return False
            elif among_var == 2:
                if not self.slice_from(u'e'):
                    return False
            elif among_var == 3:
                if not self.slice_from(u'i'):
                    return False
            elif among_var == 4:
                if not self.slice_from(u'o'):
                    return False
            elif among_var == 5:
                if not self.slice_from(u'u'):
                    return False
            elif among_var == 6:
                if not self.slice_from(u'.'):
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