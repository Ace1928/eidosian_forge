import operator
from . import libmp
from .libmp.backend import basestring
from .libmp import (
from .matrices.matrices import _matrix
from .ctx_base import StandardBaseContext
class ivmpf_constant(ivmpf):

    def __new__(cls, f):
        self = new(cls)
        self._f = f
        return self

    def _get_mpi_(self):
        prec = self.ctx._prec[0]
        a = self._f(prec, round_floor)
        b = self._f(prec, round_ceiling)
        return (a, b)
    _mpi_ = property(_get_mpi_)