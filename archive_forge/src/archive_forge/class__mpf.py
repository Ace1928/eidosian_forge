from .libmp.backend import basestring, exec_
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import rational
from . import function_docs
class _mpf(mpnumeric):
    """
    An mpf instance holds a real-valued floating-point number. mpf:s
    work analogously to Python floats, but support arbitrary-precision
    arithmetic.
    """
    __slots__ = ['_mpf_']

    def __new__(cls, val=fzero, **kwargs):
        """A new mpf can be created from a Python float, an int, a
        or a decimal string representing a number in floating-point
        format."""
        prec, rounding = cls.context._prec_rounding
        if kwargs:
            prec = kwargs.get('prec', prec)
            if 'dps' in kwargs:
                prec = dps_to_prec(kwargs['dps'])
            rounding = kwargs.get('rounding', rounding)
        if type(val) is cls:
            sign, man, exp, bc = val._mpf_
            if not man and exp:
                return val
            v = new(cls)
            v._mpf_ = normalize(sign, man, exp, bc, prec, rounding)
            return v
        elif type(val) is tuple:
            if len(val) == 2:
                v = new(cls)
                v._mpf_ = from_man_exp(val[0], val[1], prec, rounding)
                return v
            if len(val) == 4:
                if val not in (finf, fninf, fnan):
                    sign, man, exp, bc = val
                    val = normalize(sign, MPZ(man), exp, bc, prec, rounding)
                v = new(cls)
                v._mpf_ = val
                return v
            raise ValueError
        else:
            v = new(cls)
            v._mpf_ = mpf_pos(cls.mpf_convert_arg(val, prec, rounding), prec, rounding)
            return v

    @classmethod
    def mpf_convert_arg(cls, x, prec, rounding):
        if isinstance(x, int_types):
            return from_int(x)
        if isinstance(x, float):
            return from_float(x)
        if isinstance(x, basestring):
            return from_str(x, prec, rounding)
        if isinstance(x, cls.context.constant):
            return x.func(prec, rounding)
        if hasattr(x, '_mpf_'):
            return x._mpf_
        if hasattr(x, '_mpmath_'):
            t = cls.context.convert(x._mpmath_(prec, rounding))
            if hasattr(t, '_mpf_'):
                return t._mpf_
        if hasattr(x, '_mpi_'):
            a, b = x._mpi_
            if a == b:
                return a
            raise ValueError('can only create mpf from zero-width interval')
        raise TypeError('cannot create mpf from ' + repr(x))

    @classmethod
    def mpf_convert_rhs(cls, x):
        if isinstance(x, int_types):
            return from_int(x)
        if isinstance(x, float):
            return from_float(x)
        if isinstance(x, complex_types):
            return cls.context.mpc(x)
        if isinstance(x, rational.mpq):
            p, q = x._mpq_
            return from_rational(p, q, cls.context.prec)
        if hasattr(x, '_mpf_'):
            return x._mpf_
        if hasattr(x, '_mpmath_'):
            t = cls.context.convert(x._mpmath_(*cls.context._prec_rounding))
            if hasattr(t, '_mpf_'):
                return t._mpf_
            return t
        return NotImplemented

    @classmethod
    def mpf_convert_lhs(cls, x):
        x = cls.mpf_convert_rhs(x)
        if type(x) is tuple:
            return cls.context.make_mpf(x)
        return x
    man_exp = property(lambda self: self._mpf_[1:3])
    man = property(lambda self: self._mpf_[1])
    exp = property(lambda self: self._mpf_[2])
    bc = property(lambda self: self._mpf_[3])
    real = property(lambda self: self)
    imag = property(lambda self: self.context.zero)
    conjugate = lambda self: self

    def __getstate__(self):
        return to_pickable(self._mpf_)

    def __setstate__(self, val):
        self._mpf_ = from_pickable(val)

    def __repr__(s):
        if s.context.pretty:
            return str(s)
        return "mpf('%s')" % to_str(s._mpf_, s.context._repr_digits)

    def __str__(s):
        return to_str(s._mpf_, s.context._str_digits)

    def __hash__(s):
        return mpf_hash(s._mpf_)

    def __int__(s):
        return int(to_int(s._mpf_))

    def __long__(s):
        return long(to_int(s._mpf_))

    def __float__(s):
        return to_float(s._mpf_, rnd=s.context._prec_rounding[1])

    def __complex__(s):
        return complex(float(s))

    def __nonzero__(s):
        return s._mpf_ != fzero
    __bool__ = __nonzero__

    def __abs__(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpf_ = mpf_abs(s._mpf_, prec, rounding)
        return v

    def __pos__(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpf_ = mpf_pos(s._mpf_, prec, rounding)
        return v

    def __neg__(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpf_ = mpf_neg(s._mpf_, prec, rounding)
        return v

    def _cmp(s, t, func):
        if hasattr(t, '_mpf_'):
            t = t._mpf_
        else:
            t = s.mpf_convert_rhs(t)
            if t is NotImplemented:
                return t
        return func(s._mpf_, t)

    def __cmp__(s, t):
        return s._cmp(t, mpf_cmp)

    def __lt__(s, t):
        return s._cmp(t, mpf_lt)

    def __gt__(s, t):
        return s._cmp(t, mpf_gt)

    def __le__(s, t):
        return s._cmp(t, mpf_le)

    def __ge__(s, t):
        return s._cmp(t, mpf_ge)

    def __ne__(s, t):
        v = s.__eq__(t)
        if v is NotImplemented:
            return v
        return not v

    def __rsub__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if type(t) in int_types:
            v = new(cls)
            v._mpf_ = mpf_sub(from_int(t), s._mpf_, prec, rounding)
            return v
        t = s.mpf_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t - s

    def __rdiv__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if isinstance(t, int_types):
            v = new(cls)
            v._mpf_ = mpf_rdiv_int(t, s._mpf_, prec, rounding)
            return v
        t = s.mpf_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t / s

    def __rpow__(s, t):
        t = s.mpf_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t ** s

    def __rmod__(s, t):
        t = s.mpf_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t % s

    def sqrt(s):
        return s.context.sqrt(s)

    def ae(s, t, rel_eps=None, abs_eps=None):
        return s.context.almosteq(s, t, rel_eps, abs_eps)

    def to_fixed(self, prec):
        return to_fixed(self._mpf_, prec)

    def __round__(self, *args):
        return round(float(self), *args)