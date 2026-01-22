from .basestemmer import BaseStemmer
from .among import Among
def __r_R1plus3(self):
    if not self.I_p1 <= self.cursor + 3:
        return False
    return True