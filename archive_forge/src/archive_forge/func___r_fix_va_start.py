from .basestemmer import BaseStemmer
from .among import Among
def __r_fix_va_start(self):
    try:
        v_1 = self.cursor
        try:
            v_2 = self.cursor
            v_3 = self.cursor
            try:
                if not self.eq_s(u'வோ'):
                    self.cursor = v_3
                    raise lab2()
            except lab2:
                pass
            self.cursor = v_2
            self.bra = self.cursor
            if not self.eq_s(u'வோ'):
                raise lab1()
            self.ket = self.cursor
            if not self.slice_from(u'ஓ'):
                return False
            raise lab0()
        except lab1:
            pass
        self.cursor = v_1
        try:
            v_4 = self.cursor
            v_5 = self.cursor
            try:
                if not self.eq_s(u'வொ'):
                    self.cursor = v_5
                    raise lab4()
            except lab4:
                pass
            self.cursor = v_4
            self.bra = self.cursor
            if not self.eq_s(u'வொ'):
                raise lab3()
            self.ket = self.cursor
            if not self.slice_from(u'ஒ'):
                return False
            raise lab0()
        except lab3:
            pass
        self.cursor = v_1
        try:
            v_6 = self.cursor
            v_7 = self.cursor
            try:
                if not self.eq_s(u'வு'):
                    self.cursor = v_7
                    raise lab6()
            except lab6:
                pass
            self.cursor = v_6
            self.bra = self.cursor
            if not self.eq_s(u'வு'):
                raise lab5()
            self.ket = self.cursor
            if not self.slice_from(u'உ'):
                return False
            raise lab0()
        except lab5:
            pass
        self.cursor = v_1
        v_8 = self.cursor
        v_9 = self.cursor
        try:
            if not self.eq_s(u'வூ'):
                self.cursor = v_9
                raise lab7()
        except lab7:
            pass
        self.cursor = v_8
        self.bra = self.cursor
        if not self.eq_s(u'வூ'):
            return False
        self.ket = self.cursor
        if not self.slice_from(u'ஊ'):
            return False
    except lab0:
        pass
    return True