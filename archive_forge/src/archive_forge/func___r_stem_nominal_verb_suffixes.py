from .basestemmer import BaseStemmer
from .among import Among
def __r_stem_nominal_verb_suffixes(self):
    self.ket = self.cursor
    self.B_continue_stemming_noun_suffixes = True
    try:
        v_1 = self.limit - self.cursor
        try:
            try:
                v_2 = self.limit - self.cursor
                try:
                    if not self.__r_mark_ymUs_():
                        raise lab3()
                    raise lab2()
                except lab3:
                    pass
                self.cursor = self.limit - v_2
                try:
                    if not self.__r_mark_yDU():
                        raise lab4()
                    raise lab2()
                except lab4:
                    pass
                self.cursor = self.limit - v_2
                try:
                    if not self.__r_mark_ysA():
                        raise lab5()
                    raise lab2()
                except lab5:
                    pass
                self.cursor = self.limit - v_2
                if not self.__r_mark_yken():
                    raise lab1()
            except lab2:
                pass
            raise lab0()
        except lab1:
            pass
        self.cursor = self.limit - v_1
        try:
            if not self.__r_mark_cAsInA():
                raise lab6()
            try:
                v_3 = self.limit - self.cursor
                try:
                    if not self.__r_mark_sUnUz():
                        raise lab8()
                    raise lab7()
                except lab8:
                    pass
                self.cursor = self.limit - v_3
                try:
                    if not self.__r_mark_lAr():
                        raise lab9()
                    raise lab7()
                except lab9:
                    pass
                self.cursor = self.limit - v_3
                try:
                    if not self.__r_mark_yUm():
                        raise lab10()
                    raise lab7()
                except lab10:
                    pass
                self.cursor = self.limit - v_3
                try:
                    if not self.__r_mark_sUn():
                        raise lab11()
                    raise lab7()
                except lab11:
                    pass
                self.cursor = self.limit - v_3
                try:
                    if not self.__r_mark_yUz():
                        raise lab12()
                    raise lab7()
                except lab12:
                    pass
                self.cursor = self.limit - v_3
            except lab7:
                pass
            if not self.__r_mark_ymUs_():
                raise lab6()
            raise lab0()
        except lab6:
            pass
        self.cursor = self.limit - v_1
        try:
            if not self.__r_mark_lAr():
                raise lab13()
            self.bra = self.cursor
            if not self.slice_del():
                return False
            v_4 = self.limit - self.cursor
            try:
                self.ket = self.cursor
                try:
                    v_5 = self.limit - self.cursor
                    try:
                        if not self.__r_mark_DUr():
                            raise lab16()
                        raise lab15()
                    except lab16:
                        pass
                    self.cursor = self.limit - v_5
                    try:
                        if not self.__r_mark_yDU():
                            raise lab17()
                        raise lab15()
                    except lab17:
                        pass
                    self.cursor = self.limit - v_5
                    try:
                        if not self.__r_mark_ysA():
                            raise lab18()
                        raise lab15()
                    except lab18:
                        pass
                    self.cursor = self.limit - v_5
                    if not self.__r_mark_ymUs_():
                        self.cursor = self.limit - v_4
                        raise lab14()
                except lab15:
                    pass
            except lab14:
                pass
            self.B_continue_stemming_noun_suffixes = False
            raise lab0()
        except lab13:
            pass
        self.cursor = self.limit - v_1
        try:
            if not self.__r_mark_nUz():
                raise lab19()
            try:
                v_6 = self.limit - self.cursor
                try:
                    if not self.__r_mark_yDU():
                        raise lab21()
                    raise lab20()
                except lab21:
                    pass
                self.cursor = self.limit - v_6
                if not self.__r_mark_ysA():
                    raise lab19()
            except lab20:
                pass
            raise lab0()
        except lab19:
            pass
        self.cursor = self.limit - v_1
        try:
            try:
                v_7 = self.limit - self.cursor
                try:
                    if not self.__r_mark_sUnUz():
                        raise lab24()
                    raise lab23()
                except lab24:
                    pass
                self.cursor = self.limit - v_7
                try:
                    if not self.__r_mark_yUz():
                        raise lab25()
                    raise lab23()
                except lab25:
                    pass
                self.cursor = self.limit - v_7
                try:
                    if not self.__r_mark_sUn():
                        raise lab26()
                    raise lab23()
                except lab26:
                    pass
                self.cursor = self.limit - v_7
                if not self.__r_mark_yUm():
                    raise lab22()
            except lab23:
                pass
            self.bra = self.cursor
            if not self.slice_del():
                return False
            v_8 = self.limit - self.cursor
            try:
                self.ket = self.cursor
                if not self.__r_mark_ymUs_():
                    self.cursor = self.limit - v_8
                    raise lab27()
            except lab27:
                pass
            raise lab0()
        except lab22:
            pass
        self.cursor = self.limit - v_1
        if not self.__r_mark_DUr():
            return False
        self.bra = self.cursor
        if not self.slice_del():
            return False
        v_9 = self.limit - self.cursor
        try:
            self.ket = self.cursor
            try:
                v_10 = self.limit - self.cursor
                try:
                    if not self.__r_mark_sUnUz():
                        raise lab30()
                    raise lab29()
                except lab30:
                    pass
                self.cursor = self.limit - v_10
                try:
                    if not self.__r_mark_lAr():
                        raise lab31()
                    raise lab29()
                except lab31:
                    pass
                self.cursor = self.limit - v_10
                try:
                    if not self.__r_mark_yUm():
                        raise lab32()
                    raise lab29()
                except lab32:
                    pass
                self.cursor = self.limit - v_10
                try:
                    if not self.__r_mark_sUn():
                        raise lab33()
                    raise lab29()
                except lab33:
                    pass
                self.cursor = self.limit - v_10
                try:
                    if not self.__r_mark_yUz():
                        raise lab34()
                    raise lab29()
                except lab34:
                    pass
                self.cursor = self.limit - v_10
            except lab29:
                pass
            if not self.__r_mark_ymUs_():
                self.cursor = self.limit - v_9
                raise lab28()
        except lab28:
            pass
    except lab0:
        pass
    self.bra = self.cursor
    if not self.slice_del():
        return False
    return True