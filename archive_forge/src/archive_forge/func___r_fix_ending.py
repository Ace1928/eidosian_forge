from .basestemmer import BaseStemmer
from .among import Among
def __r_fix_ending(self):
    if not len(self.current) > 3:
        return False
    self.limit_backward = self.cursor
    self.cursor = self.limit
    try:
        v_1 = self.limit - self.cursor
        try:
            self.ket = self.cursor
            if self.find_among_b(TamilStemmer.a_1) == 0:
                raise lab1()
            self.bra = self.cursor
            if not self.slice_del():
                return False
            raise lab0()
        except lab1:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            if not self.eq_s_b(u'ய்'):
                raise lab2()
            v_2 = self.limit - self.cursor
            if self.find_among_b(TamilStemmer.a_2) == 0:
                raise lab2()
            self.cursor = self.limit - v_2
            self.bra = self.cursor
            if not self.slice_del():
                return False
            raise lab0()
        except lab2:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            try:
                v_3 = self.limit - self.cursor
                try:
                    if not self.eq_s_b(u'ட்ப்'):
                        raise lab5()
                    raise lab4()
                except lab5:
                    pass
                self.cursor = self.limit - v_3
                if not self.eq_s_b(u'ட்க்'):
                    raise lab3()
            except lab4:
                pass
            self.bra = self.cursor
            if not self.slice_from(u'ள்'):
                return False
            raise lab0()
        except lab3:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            if not self.eq_s_b(u'ன்ற்'):
                raise lab6()
            self.bra = self.cursor
            if not self.slice_from(u'ல்'):
                return False
            raise lab0()
        except lab6:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            if not self.eq_s_b(u'ற்க்'):
                raise lab7()
            self.bra = self.cursor
            if not self.slice_from(u'ல்'):
                return False
            raise lab0()
        except lab7:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            if not self.eq_s_b(u'ட்ட்'):
                raise lab8()
            self.bra = self.cursor
            if not self.slice_from(u'டு'):
                return False
            raise lab0()
        except lab8:
            pass
        self.cursor = self.limit - v_1
        try:
            if not self.B_found_vetrumai_urupu:
                raise lab9()
            self.ket = self.cursor
            if not self.eq_s_b(u'த்த்'):
                raise lab9()
            v_4 = self.limit - self.cursor
            v_5 = self.limit - self.cursor
            try:
                if not self.eq_s_b(u'ை'):
                    raise lab10()
                raise lab9()
            except lab10:
                pass
            self.cursor = self.limit - v_5
            self.cursor = self.limit - v_4
            self.bra = self.cursor
            if not self.slice_from(u'ம்'):
                return False
            self.bra = self.cursor
            raise lab0()
        except lab9:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            try:
                v_6 = self.limit - self.cursor
                try:
                    if not self.eq_s_b(u'ுக்'):
                        raise lab13()
                    raise lab12()
                except lab13:
                    pass
                self.cursor = self.limit - v_6
                if not self.eq_s_b(u'ுக்க்'):
                    raise lab11()
            except lab12:
                pass
            self.bra = self.cursor
            if not self.slice_from(u'்'):
                return False
            raise lab0()
        except lab11:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            if not self.eq_s_b(u'்'):
                raise lab14()
            if self.find_among_b(TamilStemmer.a_3) == 0:
                raise lab14()
            if not self.eq_s_b(u'்'):
                raise lab14()
            if self.find_among_b(TamilStemmer.a_4) == 0:
                raise lab14()
            self.bra = self.cursor
            if not self.slice_del():
                return False
            raise lab0()
        except lab14:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            if not self.eq_s_b(u'ுக்'):
                raise lab15()
            self.bra = self.cursor
            if not self.slice_from(u'்'):
                return False
            raise lab0()
        except lab15:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            if not self.eq_s_b(u'்'):
                raise lab16()
            if self.find_among_b(TamilStemmer.a_5) == 0:
                raise lab16()
            self.bra = self.cursor
            if not self.slice_del():
                return False
            raise lab0()
        except lab16:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            if not self.eq_s_b(u'்'):
                raise lab17()
            try:
                v_7 = self.limit - self.cursor
                try:
                    if self.find_among_b(TamilStemmer.a_6) == 0:
                        raise lab19()
                    raise lab18()
                except lab19:
                    pass
                self.cursor = self.limit - v_7
                if self.find_among_b(TamilStemmer.a_7) == 0:
                    raise lab17()
            except lab18:
                pass
            if not self.eq_s_b(u'்'):
                raise lab17()
            self.bra = self.cursor
            if not self.slice_from(u'்'):
                return False
            raise lab0()
        except lab17:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            if self.find_among_b(TamilStemmer.a_8) == 0:
                raise lab20()
            self.bra = self.cursor
            if not self.slice_del():
                return False
            raise lab0()
        except lab20:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            if not self.eq_s_b(u'னு'):
                raise lab21()
            v_8 = self.limit - self.cursor
            v_9 = self.limit - self.cursor
            try:
                if self.find_among_b(TamilStemmer.a_9) == 0:
                    raise lab22()
                raise lab21()
            except lab22:
                pass
            self.cursor = self.limit - v_9
            self.cursor = self.limit - v_8
            self.bra = self.cursor
            if not self.slice_del():
                return False
            raise lab0()
        except lab21:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            if not self.eq_s_b(u'ங்'):
                raise lab23()
            v_10 = self.limit - self.cursor
            v_11 = self.limit - self.cursor
            try:
                if not self.eq_s_b(u'ை'):
                    raise lab24()
                raise lab23()
            except lab24:
                pass
            self.cursor = self.limit - v_11
            self.cursor = self.limit - v_10
            self.bra = self.cursor
            if not self.slice_from(u'ம்'):
                return False
            raise lab0()
        except lab23:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            if not self.eq_s_b(u'ங்'):
                raise lab25()
            self.bra = self.cursor
            if not self.slice_del():
                return False
            raise lab0()
        except lab25:
            pass
        self.cursor = self.limit - v_1
        self.ket = self.cursor
        if not self.eq_s_b(u'்'):
            return False
        v_12 = self.limit - self.cursor
        try:
            v_13 = self.limit - self.cursor
            try:
                if self.find_among_b(TamilStemmer.a_10) == 0:
                    raise lab27()
                raise lab26()
            except lab27:
                pass
            self.cursor = self.limit - v_13
            if not self.eq_s_b(u'்'):
                return False
        except lab26:
            pass
        self.cursor = self.limit - v_12
        self.bra = self.cursor
        if not self.slice_del():
            return False
    except lab0:
        pass
    self.cursor = self.limit_backward
    return True