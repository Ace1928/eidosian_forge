from . import matrix
from .polynomial import Polynomial
from ..pari import pari
def _is_non_trivial(self, N):
    for h in self.H2_class:
        if h % N != 0:
            return True
    return False