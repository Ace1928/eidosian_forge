import operator
from . import libmp
from .libmp.backend import basestring
from .libmp import (
from .matrices.matrices import _matrix
from .ctx_base import StandardBaseContext
class ivmpf(object):
    """
    Interval arithmetic class. Precision is controlled by iv.prec.
    """

    def __new__(cls, x=0):
        return cls.ctx.convert(x)

    def cast(self, cls, f_convert):
        a, b = self._mpi_
        if a == b:
            return cls(f_convert(a))
        raise ValueError

    def __int__(self):
        return self.cast(int, libmp.to_int)

    def __float__(self):
        return self.cast(float, libmp.to_float)

    def __complex__(self):
        return self.cast(complex, libmp.to_float)

    def __hash__(self):
        a, b = self._mpi_
        if a == b:
            return mpf_hash(a)
        else:
            return hash(self._mpi_)

    @property
    def real(self):
        return self

    @property
    def imag(self):
        return self.ctx.zero

    def conjugate(self):
        return self

    @property
    def a(self):
        a, b = self._mpi_
        return self.ctx.make_mpf((a, a))

    @property
    def b(self):
        a, b = self._mpi_
        return self.ctx.make_mpf((b, b))

    @property
    def mid(self):
        ctx = self.ctx
        v = mpi_mid(self._mpi_, ctx.prec)
        return ctx.make_mpf((v, v))

    @property
    def delta(self):
        ctx = self.ctx
        v = mpi_delta(self._mpi_, ctx.prec)
        return ctx.make_mpf((v, v))

    @property
    def _mpci_(self):
        return (self._mpi_, mpi_zero)

    def _compare(*args):
        raise TypeError('no ordering relation is defined for intervals')
    __gt__ = _compare
    __le__ = _compare
    __gt__ = _compare
    __ge__ = _compare

    def __contains__(self, t):
        t = self.ctx.mpf(t)
        return self.a <= t.a and t.b <= self.b

    def __str__(self):
        return mpi_str(self._mpi_, self.ctx.prec)

    def __repr__(self):
        if self.ctx.pretty:
            return str(self)
        a, b = self._mpi_
        n = repr_dps(self.ctx.prec)
        a = libmp.to_str(a, n)
        b = libmp.to_str(b, n)
        return 'mpi(%r, %r)' % (a, b)

    def _compare(s, t, cmpfun):
        if not hasattr(t, '_mpi_'):
            try:
                t = s.ctx.convert(t)
            except:
                return NotImplemented
        return cmpfun(s._mpi_, t._mpi_)

    def __eq__(s, t):
        return s._compare(t, libmp.mpi_eq)

    def __ne__(s, t):
        return s._compare(t, libmp.mpi_ne)

    def __lt__(s, t):
        return s._compare(t, libmp.mpi_lt)

    def __le__(s, t):
        return s._compare(t, libmp.mpi_le)

    def __gt__(s, t):
        return s._compare(t, libmp.mpi_gt)

    def __ge__(s, t):
        return s._compare(t, libmp.mpi_ge)

    def __abs__(self):
        return self.ctx.make_mpf(mpi_abs(self._mpi_, self.ctx.prec))

    def __pos__(self):
        return self.ctx.make_mpf(mpi_pos(self._mpi_, self.ctx.prec))

    def __neg__(self):
        return self.ctx.make_mpf(mpi_neg(self._mpi_, self.ctx.prec))

    def ae(s, t, rel_eps=None, abs_eps=None):
        return s.ctx.almosteq(s, t, rel_eps, abs_eps)