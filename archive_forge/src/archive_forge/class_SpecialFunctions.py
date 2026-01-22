from ..libmp.backend import xrange
import math
import cmath
class SpecialFunctions(object):
    """
    This class implements special functions using high-level code.

    Elementary and some other functions (e.g. gamma function, basecase
    hypergeometric series) are assumed to be predefined by the context as
    "builtins" or "low-level" functions.
    """
    defined_functions = {}
    THETA_Q_LIM = 1 - 10 ** (-7)

    def __init__(self):
        cls = self.__class__
        for name in cls.defined_functions:
            f, wrap = cls.defined_functions[name]
            cls._wrap_specfun(name, f, wrap)
        self.mpq_1 = self._mpq((1, 1))
        self.mpq_0 = self._mpq((0, 1))
        self.mpq_1_2 = self._mpq((1, 2))
        self.mpq_3_2 = self._mpq((3, 2))
        self.mpq_1_4 = self._mpq((1, 4))
        self.mpq_1_16 = self._mpq((1, 16))
        self.mpq_3_16 = self._mpq((3, 16))
        self.mpq_5_2 = self._mpq((5, 2))
        self.mpq_3_4 = self._mpq((3, 4))
        self.mpq_7_4 = self._mpq((7, 4))
        self.mpq_5_4 = self._mpq((5, 4))
        self.mpq_1_3 = self._mpq((1, 3))
        self.mpq_2_3 = self._mpq((2, 3))
        self.mpq_4_3 = self._mpq((4, 3))
        self.mpq_1_6 = self._mpq((1, 6))
        self.mpq_5_6 = self._mpq((5, 6))
        self.mpq_5_3 = self._mpq((5, 3))
        self._misc_const_cache = {}
        self._aliases.update({'phase': 'arg', 'conjugate': 'conj', 'nthroot': 'root', 'polygamma': 'psi', 'hurwitz': 'zeta', 'fibonacci': 'fib', 'factorial': 'fac'})
        self.zetazero_memoized = self.memoize(self.zetazero)

    @classmethod
    def _wrap_specfun(cls, name, f, wrap):
        setattr(cls, name, f)

    def _besselj(ctx, n, z):
        raise NotImplementedError

    def _erf(ctx, z):
        raise NotImplementedError

    def _erfc(ctx, z):
        raise NotImplementedError

    def _gamma_upper_int(ctx, z, a):
        raise NotImplementedError

    def _expint_int(ctx, n, z):
        raise NotImplementedError

    def _zeta(ctx, s):
        raise NotImplementedError

    def _zetasum_fast(ctx, s, a, n, derivatives, reflect):
        raise NotImplementedError

    def _ei(ctx, z):
        raise NotImplementedError

    def _e1(ctx, z):
        raise NotImplementedError

    def _ci(ctx, z):
        raise NotImplementedError

    def _si(ctx, z):
        raise NotImplementedError

    def _altzeta(ctx, s):
        raise NotImplementedError