import string
from ..sage_helper import _within_sage, sage_method
def _exponents_of_word(self, word):
    exponents = self.U * abelianize_word(word, self.domain_gens)
    return self._normalize_exponents(exponents)