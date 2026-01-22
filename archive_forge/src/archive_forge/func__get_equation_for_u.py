from . import matrix
from .polynomial import Polynomial
from ..pari import pari
def _get_equation_for_u(self, N):
    if self._is_non_trivial(N):
        if N == 2:
            return (2, [])
        else:
            cyclo = Polynomial.parse_string(str(pari.polcyclo(N, 'u')))
            return (N, [cyclo])
    else:
        return (1, [])