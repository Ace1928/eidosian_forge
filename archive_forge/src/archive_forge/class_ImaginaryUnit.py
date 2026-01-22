from __future__ import annotations
import numbers
import decimal
import fractions
import math
import re as regex
import sys
from functools import lru_cache
from .containers import Tuple
from .sympify import (SympifyError, _sympy_converter, sympify, _convert_numpy_types,
from .singleton import S, Singleton
from .basic import Basic
from .expr import Expr, AtomicExpr
from .evalf import pure_complex
from .cache import cacheit, clear_cache
from .decorators import _sympifyit
from .logic import fuzzy_not
from .kind import NumberKind
from sympy.external.gmpy import SYMPY_INTS, HAS_GMPY, gmpy
from sympy.multipledispatch import dispatch
import mpmath
import mpmath.libmp as mlib
from mpmath.libmp import bitcount, round_nearest as rnd
from mpmath.libmp.backend import MPZ
from mpmath.libmp import mpf_pow, mpf_pi, mpf_e, phi_fixed
from mpmath.ctx_mp import mpnumeric
from mpmath.libmp.libmpf import (
from sympy.utilities.misc import as_int, debug, filldedent
from .parameters import global_parameters
from .power import Pow, integer_nthroot
from .mul import Mul
from .add import Add
class ImaginaryUnit(AtomicExpr, metaclass=Singleton):
    """The imaginary unit, `i = \\sqrt{-1}`.

    I is a singleton, and can be accessed by ``S.I``, or can be
    imported as ``I``.

    Examples
    ========

    >>> from sympy import I, sqrt
    >>> sqrt(-1)
    I
    >>> I*I
    -1
    >>> 1/I
    -I

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Imaginary_unit
    """
    is_commutative = True
    is_imaginary = True
    is_finite = True
    is_number = True
    is_algebraic = True
    is_transcendental = False
    kind = NumberKind
    __slots__ = ()

    def _latex(self, printer):
        return printer._settings['imaginary_unit_latex']

    @staticmethod
    def __abs__():
        return S.One

    def _eval_evalf(self, prec):
        return self

    def _eval_conjugate(self):
        return -S.ImaginaryUnit

    def _eval_power(self, expt):
        """
        b is I = sqrt(-1)
        e is symbolic object but not equal to 0, 1

        I**r -> (-1)**(r/2) -> exp(r/2*Pi*I) -> sin(Pi*r/2) + cos(Pi*r/2)*I, r is decimal
        I**0 mod 4 -> 1
        I**1 mod 4 -> I
        I**2 mod 4 -> -1
        I**3 mod 4 -> -I
        """
        if isinstance(expt, Integer):
            expt = expt % 4
            if expt == 0:
                return S.One
            elif expt == 1:
                return S.ImaginaryUnit
            elif expt == 2:
                return S.NegativeOne
            elif expt == 3:
                return -S.ImaginaryUnit
        if isinstance(expt, Rational):
            i, r = divmod(expt, 2)
            rv = Pow(S.ImaginaryUnit, r, evaluate=False)
            if i % 2:
                return Mul(S.NegativeOne, rv, evaluate=False)
            return rv

    def as_base_exp(self):
        return (S.NegativeOne, S.Half)

    @property
    def _mpc_(self):
        return (Float(0)._mpf_, Float(1)._mpf_)