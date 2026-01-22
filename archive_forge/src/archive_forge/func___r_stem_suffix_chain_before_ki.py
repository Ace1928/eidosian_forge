from .basestemmer import BaseStemmer
from .among import Among
def __r_stem_suffix_chain_before_ki(self):
    self.ket = self.cursor
    if not self.__r_mark_ki():
        return False
    try:
        v_1 = self.limit - self.cursor
        try:
            if not self.__r_mark_DA():
                raise lab1()
            self.bra = self.cursor
            if not self.slice_del():
                return False
            v_2 = self.limit - self.cursor
            try:
                self.ket = self.cursor
                try:
                    v_3 = self.limit - self.cursor
                    try:
                        if not self.__r_mark_lAr():
                            raise lab4()
                        self.bra = self.cursor
                        if not self.slice_del():
                            return False
                        v_4 = self.limit - self.cursor
                        try:
                            if not self.__r_stem_suffix_chain_before_ki():
                                self.cursor = self.limit - v_4
                                raise lab5()
                        except lab5:
                            pass
                        raise lab3()
                    except lab4:
                        pass
                    self.cursor = self.limit - v_3
                    if not self.__r_mark_possessives():
                        self.cursor = self.limit - v_2
                        raise lab2()
                    self.bra = self.cursor
                    if not self.slice_del():
                        return False
                    v_5 = self.limit - self.cursor
                    try:
                        self.ket = self.cursor
                        if not self.__r_mark_lAr():
                            self.cursor = self.limit - v_5
                            raise lab6()
                        self.bra = self.cursor
                        if not self.slice_del():
                            return False
                        if not self.__r_stem_suffix_chain_before_ki():
                            self.cursor = self.limit - v_5
                            raise lab6()
                    except lab6:
                        pass
                except lab3:
                    pass
            except lab2:
                pass
            raise lab0()
        except lab1:
            pass
        self.cursor = self.limit - v_1
        try:
            if not self.__r_mark_nUn():
                raise lab7()
            self.bra = self.cursor
            if not self.slice_del():
                return False
            v_6 = self.limit - self.cursor
            try:
                self.ket = self.cursor
                try:
                    v_7 = self.limit - self.cursor
                    try:
                        if not self.__r_mark_lArI():
                            raise lab10()
                        self.bra = self.cursor
                        if not self.slice_del():
                            return False
                        raise lab9()
                    except lab10:
                        pass
                    self.cursor = self.limit - v_7
                    try:
                        self.ket = self.cursor
                        try:
                            v_8 = self.limit - self.cursor
                            try:
                                if not self.__r_mark_possessives():
                                    raise lab13()
                                raise lab12()
                            except lab13:
                                pass
                            self.cursor = self.limit - v_8
                            if not self.__r_mark_sU():
                                raise lab11()
                        except lab12:
                            pass
                        self.bra = self.cursor
                        if not self.slice_del():
                            return False
                        v_9 = self.limit - self.cursor
                        try:
                            self.ket = self.cursor
                            if not self.__r_mark_lAr():
                                self.cursor = self.limit - v_9
                                raise lab14()
                            self.bra = self.cursor
                            if not self.slice_del():
                                return False
                            if not self.__r_stem_suffix_chain_before_ki():
                                self.cursor = self.limit - v_9
                                raise lab14()
                        except lab14:
                            pass
                        raise lab9()
                    except lab11:
                        pass
                    self.cursor = self.limit - v_7
                    if not self.__r_stem_suffix_chain_before_ki():
                        self.cursor = self.limit - v_6
                        raise lab8()
                except lab9:
                    pass
            except lab8:
                pass
            raise lab0()
        except lab7:
            pass
        self.cursor = self.limit - v_1
        if not self.__r_mark_ndA():
            return False
        try:
            v_10 = self.limit - self.cursor
            try:
                if not self.__r_mark_lArI():
                    raise lab16()
                self.bra = self.cursor
                if not self.slice_del():
                    return False
                raise lab15()
            except lab16:
                pass
            self.cursor = self.limit - v_10
            try:
                if not self.__r_mark_sU():
                    raise lab17()
                self.bra = self.cursor
                if not self.slice_del():
                    return False
                v_11 = self.limit - self.cursor
                try:
                    self.ket = self.cursor
                    if not self.__r_mark_lAr():
                        self.cursor = self.limit - v_11
                        raise lab18()
                    self.bra = self.cursor
                    if not self.slice_del():
                        return False
                    if not self.__r_stem_suffix_chain_before_ki():
                        self.cursor = self.limit - v_11
                        raise lab18()
                except lab18:
                    pass
                raise lab15()
            except lab17:
                pass
            self.cursor = self.limit - v_10
            if not self.__r_stem_suffix_chain_before_ki():
                return False
        except lab15:
            pass
    except lab0:
        pass
    return True