from .basestemmer import BaseStemmer
from .among import Among
def __r_remove_tense_suffixes(self):
    self.B_found_a_match = True
    while True:
        v_1 = self.cursor
        try:
            if not self.B_found_a_match:
                raise lab0()
            v_2 = self.cursor
            self.__r_remove_tense_suffix()
            self.cursor = v_2
            continue
        except lab0:
            pass
        self.cursor = v_1
        break
    return True