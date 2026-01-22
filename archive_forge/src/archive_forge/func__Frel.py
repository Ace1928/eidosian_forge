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
def _Frel(self, other, op):
    try:
        other = _sympify(other)
    except SympifyError:
        return NotImplemented
    if other.is_Rational:
        '\n            >>> f = Float(.1,2)\n            >>> i = 1234567890\n            >>> (f*i)._mpf_\n            (0, 471, 18, 9)\n            >>> mlib.mpf_mul(f._mpf_, mlib.from_int(i))\n            (0, 505555550955, -12, 39)\n            '
        smpf = mlib.mpf_mul(self._mpf_, mlib.from_int(other.q))
        ompf = mlib.from_int(other.p)
        return _sympify(bool(op(smpf, ompf)))
    elif other.is_Float:
        return _sympify(bool(op(self._mpf_, other._mpf_)))
    elif other.is_comparable and other not in (S.Infinity, S.NegativeInfinity):
        other = other.evalf(prec_to_dps(self._prec))
        if other._prec > 1:
            if other.is_Number:
                return _sympify(bool(op(self._mpf_, other._as_mpf_val(self._prec))))