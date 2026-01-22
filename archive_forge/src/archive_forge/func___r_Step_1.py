from .basestemmer import BaseStemmer
from .among import Among
def __r_Step_1(self):
    self.ket = self.cursor
    among_var = self.find_among_b(SerbianStemmer.a_1)
    if among_var == 0:
        return False
    self.bra = self.cursor
    if among_var == 1:
        if not self.slice_from(u'loga'):
            return False
    elif among_var == 2:
        if not self.slice_from(u'peh'):
            return False
    elif among_var == 3:
        if not self.slice_from(u'vojka'):
            return False
    elif among_var == 4:
        if not self.slice_from(u'bojka'):
            return False
    elif among_var == 5:
        if not self.slice_from(u'jak'):
            return False
    elif among_var == 6:
        if not self.slice_from(u'čajni'):
            return False
    elif among_var == 7:
        if not self.B_no_diacritics:
            return False
        if not self.slice_from(u'cajni'):
            return False
    elif among_var == 8:
        if not self.slice_from(u'erni'):
            return False
    elif among_var == 9:
        if not self.slice_from(u'larni'):
            return False
    elif among_var == 10:
        if not self.slice_from(u'esni'):
            return False
    elif among_var == 11:
        if not self.slice_from(u'anjca'):
            return False
    elif among_var == 12:
        if not self.slice_from(u'ajca'):
            return False
    elif among_var == 13:
        if not self.slice_from(u'ljca'):
            return False
    elif among_var == 14:
        if not self.slice_from(u'ejca'):
            return False
    elif among_var == 15:
        if not self.slice_from(u'ojca'):
            return False
    elif among_var == 16:
        if not self.slice_from(u'ajka'):
            return False
    elif among_var == 17:
        if not self.slice_from(u'ojka'):
            return False
    elif among_var == 18:
        if not self.slice_from(u'šca'):
            return False
    elif among_var == 19:
        if not self.slice_from(u'ing'):
            return False
    elif among_var == 20:
        if not self.slice_from(u'tvenik'):
            return False
    elif among_var == 21:
        if not self.slice_from(u'tetika'):
            return False
    elif among_var == 22:
        if not self.slice_from(u'nstva'):
            return False
    elif among_var == 23:
        if not self.slice_from(u'nik'):
            return False
    elif among_var == 24:
        if not self.slice_from(u'tik'):
            return False
    elif among_var == 25:
        if not self.slice_from(u'zik'):
            return False
    elif among_var == 26:
        if not self.slice_from(u'snik'):
            return False
    elif among_var == 27:
        if not self.slice_from(u'kusi'):
            return False
    elif among_var == 28:
        if not self.slice_from(u'kusni'):
            return False
    elif among_var == 29:
        if not self.slice_from(u'kustva'):
            return False
    elif among_var == 30:
        if not self.slice_from(u'dušni'):
            return False
    elif among_var == 31:
        if not self.B_no_diacritics:
            return False
        if not self.slice_from(u'dusni'):
            return False
    elif among_var == 32:
        if not self.slice_from(u'antni'):
            return False
    elif among_var == 33:
        if not self.slice_from(u'bilni'):
            return False
    elif among_var == 34:
        if not self.slice_from(u'tilni'):
            return False
    elif among_var == 35:
        if not self.slice_from(u'avilni'):
            return False
    elif among_var == 36:
        if not self.slice_from(u'silni'):
            return False
    elif among_var == 37:
        if not self.slice_from(u'gilni'):
            return False
    elif among_var == 38:
        if not self.slice_from(u'rilni'):
            return False
    elif among_var == 39:
        if not self.slice_from(u'nilni'):
            return False
    elif among_var == 40:
        if not self.slice_from(u'alni'):
            return False
    elif among_var == 41:
        if not self.slice_from(u'ozni'):
            return False
    elif among_var == 42:
        if not self.slice_from(u'ravi'):
            return False
    elif among_var == 43:
        if not self.slice_from(u'stavni'):
            return False
    elif among_var == 44:
        if not self.slice_from(u'pravni'):
            return False
    elif among_var == 45:
        if not self.slice_from(u'tivni'):
            return False
    elif among_var == 46:
        if not self.slice_from(u'sivni'):
            return False
    elif among_var == 47:
        if not self.slice_from(u'atni'):
            return False
    elif among_var == 48:
        if not self.slice_from(u'enta'):
            return False
    elif among_var == 49:
        if not self.slice_from(u'tetni'):
            return False
    elif among_var == 50:
        if not self.slice_from(u'pletni'):
            return False
    elif among_var == 51:
        if not self.slice_from(u'šavi'):
            return False
    elif among_var == 52:
        if not self.B_no_diacritics:
            return False
        if not self.slice_from(u'savi'):
            return False
    elif among_var == 53:
        if not self.slice_from(u'anta'):
            return False
    elif among_var == 54:
        if not self.slice_from(u'ačka'):
            return False
    elif among_var == 55:
        if not self.B_no_diacritics:
            return False
        if not self.slice_from(u'acka'):
            return False
    elif among_var == 56:
        if not self.slice_from(u'uška'):
            return False
    elif among_var == 57:
        if not self.B_no_diacritics:
            return False
        if not self.slice_from(u'uska'):
            return False
    elif among_var == 58:
        if not self.slice_from(u'atka'):
            return False
    elif among_var == 59:
        if not self.slice_from(u'etka'):
            return False
    elif among_var == 60:
        if not self.slice_from(u'itka'):
            return False
    elif among_var == 61:
        if not self.slice_from(u'otka'):
            return False
    elif among_var == 62:
        if not self.slice_from(u'utka'):
            return False
    elif among_var == 63:
        if not self.slice_from(u'eskna'):
            return False
    elif among_var == 64:
        if not self.slice_from(u'tični'):
            return False
    elif among_var == 65:
        if not self.B_no_diacritics:
            return False
        if not self.slice_from(u'ticni'):
            return False
    elif among_var == 66:
        if not self.slice_from(u'ojska'):
            return False
    elif among_var == 67:
        if not self.slice_from(u'esma'):
            return False
    elif among_var == 68:
        if not self.slice_from(u'metra'):
            return False
    elif among_var == 69:
        if not self.slice_from(u'centra'):
            return False
    elif among_var == 70:
        if not self.slice_from(u'istra'):
            return False
    elif among_var == 71:
        if not self.slice_from(u'osti'):
            return False
    elif among_var == 72:
        if not self.B_no_diacritics:
            return False
        if not self.slice_from(u'osti'):
            return False
    elif among_var == 73:
        if not self.slice_from(u'dba'):
            return False
    elif among_var == 74:
        if not self.slice_from(u'čka'):
            return False
    elif among_var == 75:
        if not self.slice_from(u'mca'):
            return False
    elif among_var == 76:
        if not self.slice_from(u'nca'):
            return False
    elif among_var == 77:
        if not self.slice_from(u'voljni'):
            return False
    elif among_var == 78:
        if not self.slice_from(u'anki'):
            return False
    elif among_var == 79:
        if not self.slice_from(u'vca'):
            return False
    elif among_var == 80:
        if not self.slice_from(u'sca'):
            return False
    elif among_var == 81:
        if not self.slice_from(u'rca'):
            return False
    elif among_var == 82:
        if not self.slice_from(u'alca'):
            return False
    elif among_var == 83:
        if not self.slice_from(u'elca'):
            return False
    elif among_var == 84:
        if not self.slice_from(u'olca'):
            return False
    elif among_var == 85:
        if not self.slice_from(u'njca'):
            return False
    elif among_var == 86:
        if not self.slice_from(u'ekta'):
            return False
    elif among_var == 87:
        if not self.slice_from(u'izma'):
            return False
    elif among_var == 88:
        if not self.slice_from(u'jebi'):
            return False
    elif among_var == 89:
        if not self.slice_from(u'baci'):
            return False
    elif among_var == 90:
        if not self.slice_from(u'ašni'):
            return False
    else:
        if not self.B_no_diacritics:
            return False
        if not self.slice_from(u'asni'):
            return False
    return True