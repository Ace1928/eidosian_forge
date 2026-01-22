from .basestemmer import BaseStemmer
from .among import Among
def __r_Step_2(self):
    self.ket = self.cursor
    among_var = self.find_among_b(EnglishStemmer.a_5)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if not self.__r_R1():
        return False
    if among_var == 1:
        if not self.slice_from(u'tion'):
            return False
    elif among_var == 2:
        if not self.slice_from(u'ence'):
            return False
    elif among_var == 3:
        if not self.slice_from(u'ance'):
            return False
    elif among_var == 4:
        if not self.slice_from(u'able'):
            return False
    elif among_var == 5:
        if not self.slice_from(u'ent'):
            return False
    elif among_var == 6:
        if not self.slice_from(u'ize'):
            return False
    elif among_var == 7:
        if not self.slice_from(u'ate'):
            return False
    elif among_var == 8:
        if not self.slice_from(u'al'):
            return False
    elif among_var == 9:
        if not self.slice_from(u'ful'):
            return False
    elif among_var == 10:
        if not self.slice_from(u'ous'):
            return False
    elif among_var == 11:
        if not self.slice_from(u'ive'):
            return False
    elif among_var == 12:
        if not self.slice_from(u'ble'):
            return False
    elif among_var == 13:
        if not self.eq_s_b(u'l'):
            return False
        if not self.slice_from(u'og'):
            return False
    elif among_var == 14:
        if not self.slice_from(u'less'):
            return False
    else:
        if not self.in_grouping_b(EnglishStemmer.g_valid_LI, 99, 116):
            return False
        if not self.slice_del():
            return False
    return True