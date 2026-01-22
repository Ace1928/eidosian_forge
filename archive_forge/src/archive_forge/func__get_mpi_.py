import operator
from . import libmp
from .libmp.backend import basestring
from .libmp import (
from .matrices.matrices import _matrix
from .ctx_base import StandardBaseContext
def _get_mpi_(self):
    prec = self.ctx._prec[0]
    a = self._f(prec, round_floor)
    b = self._f(prec, round_ceiling)
    return (a, b)