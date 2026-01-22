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
def _eval_power_exp_is_pow(self, arg):
    if arg.is_Number:
        if arg is oo:
            return oo
        elif arg == -oo:
            return S.Zero
    from sympy.functions.elementary.exponential import log
    if isinstance(arg, log):
        return arg.args[0]
    elif not arg.is_Add:
        Ioo = I * oo
        if arg in [Ioo, -Ioo]:
            return nan
        coeff = arg.coeff(pi * I)
        if coeff:
            if (2 * coeff).is_integer:
                if coeff.is_even:
                    return S.One
                elif coeff.is_odd:
                    return S.NegativeOne
                elif (coeff + S.Half).is_even:
                    return -I
                elif (coeff + S.Half).is_odd:
                    return I
            elif coeff.is_Rational:
                ncoeff = coeff % 2
                if ncoeff > 1:
                    ncoeff -= 2
                if ncoeff != coeff:
                    return S.Exp1 ** (ncoeff * S.Pi * S.ImaginaryUnit)
        coeff, terms = arg.as_coeff_Mul()
        if coeff in (oo, -oo):
            return
        coeffs, log_term = ([coeff], None)
        for term in Mul.make_args(terms):
            if isinstance(term, log):
                if log_term is None:
                    log_term = term.args[0]
                else:
                    return
            elif term.is_comparable:
                coeffs.append(term)
            else:
                return
        return log_term ** Mul(*coeffs) if log_term else None
    elif arg.is_Add:
        out = []
        add = []
        argchanged = False
        for a in arg.args:
            if a is S.One:
                add.append(a)
                continue
            newa = self ** a
            if isinstance(newa, Pow) and newa.base is self:
                if newa.exp != a:
                    add.append(newa.exp)
                    argchanged = True
                else:
                    add.append(a)
            else:
                out.append(newa)
        if out or argchanged:
            return Mul(*out) * Pow(self, Add(*add), evaluate=False)
    elif arg.is_Matrix:
        return arg.exp()