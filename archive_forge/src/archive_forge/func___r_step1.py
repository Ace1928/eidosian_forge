from .basestemmer import BaseStemmer
from .among import Among
def __r_step1(self):
    self.ket = self.cursor
    among_var = self.find_among_b(GreekStemmer.a_1)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if among_var == 1:
        if not self.slice_from(u'φα'):
            return False
    elif among_var == 2:
        if not self.slice_from(u'σκα'):
            return False
    elif among_var == 3:
        if not self.slice_from(u'ολο'):
            return False
    elif among_var == 4:
        if not self.slice_from(u'σο'):
            return False
    elif among_var == 5:
        if not self.slice_from(u'τατο'):
            return False
    elif among_var == 6:
        if not self.slice_from(u'κρε'):
            return False
    elif among_var == 7:
        if not self.slice_from(u'περ'):
            return False
    elif among_var == 8:
        if not self.slice_from(u'τερ'):
            return False
    elif among_var == 9:
        if not self.slice_from(u'φω'):
            return False
    elif among_var == 10:
        if not self.slice_from(u'καθεστ'):
            return False
    elif not self.slice_from(u'γεγον'):
        return False
    self.B_test1 = False
    return True