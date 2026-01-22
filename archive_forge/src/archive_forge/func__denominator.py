from ..sage_helper import _within_sage
from ..pari import Gen, pari
from ..math_basics import prod
def _denominator(self):
    return self._multiply_terms(-1)