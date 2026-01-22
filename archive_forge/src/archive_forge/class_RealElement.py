from sympy.polys.domains.domainelement import DomainElement
from sympy.utilities import public
from mpmath.ctx_mp_python import PythonMPContext, _mpf, _mpc, _constant
from mpmath.libmp import (MPZ_ONE, fzero, fone, finf, fninf, fnan,
from mpmath.rational import mpq
@public
class RealElement(_mpf, DomainElement):
    """An element of a real domain. """
    __slots__ = ('__mpf__',)

    def _set_mpf(self, val):
        self.__mpf__ = val
    _mpf_ = property(lambda self: self.__mpf__, _set_mpf)

    def parent(self):
        return self.context._parent