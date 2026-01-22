from __future__ import annotations
from typing import Callable
from math import log as _log, sqrt as _sqrt
from itertools import product
from .sympify import _sympify
from .cache import cacheit
from .singleton import S
from .expr import Expr
from .evalf import PrecisionExhausted
from .function import (expand_complex, expand_multinomial,
from .logic import fuzzy_bool, fuzzy_not, fuzzy_and, fuzzy_or
from .parameters import global_parameters
from .relational import is_gt, is_lt
from .kind import NumberKind, UndefinedKind
from sympy.external.gmpy import HAS_GMPY, gmpy
from sympy.utilities.iterables import sift
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.misc import as_int
from sympy.multipledispatch import Dispatcher
from mpmath.libmp import sqrtrem as mpmath_sqrtrem
from .add import Add
from .numbers import Integer
from .mul import Mul, _keep_coeff
from .symbol import Symbol, Dummy, symbols
def _eval_expand_power_base(self, **hints):
    """(a*b)**n -> a**n * b**n"""
    force = hints.get('force', False)
    b = self.base
    e = self.exp
    if not b.is_Mul:
        return self
    cargs, nc = b.args_cnc(split_1=False)
    if nc:
        nc = [i._eval_expand_power_base(**hints) if hasattr(i, '_eval_expand_power_base') else i for i in nc]
        if e.is_Integer:
            if e.is_positive:
                rv = Mul(*nc * e)
            else:
                rv = Mul(*[i ** (-1) for i in nc[::-1]] * -e)
            if cargs:
                rv *= Mul(*cargs) ** e
            return rv
        if not cargs:
            return self.func(Mul(*nc), e, evaluate=False)
        nc = [Mul(*nc)]
    other, maybe_real = sift(cargs, lambda x: x.is_extended_real is False, binary=True)

    def pred(x):
        if x is S.ImaginaryUnit:
            return S.ImaginaryUnit
        polar = x.is_polar
        if polar:
            return True
        if polar is None:
            return fuzzy_bool(x.is_extended_nonnegative)
    sifted = sift(maybe_real, pred)
    nonneg = sifted[True]
    other += sifted[None]
    neg = sifted[False]
    imag = sifted[S.ImaginaryUnit]
    if imag:
        I = S.ImaginaryUnit
        i = len(imag) % 4
        if i == 0:
            pass
        elif i == 1:
            other.append(I)
        elif i == 2:
            if neg:
                nonn = -neg.pop()
                if nonn is not S.One:
                    nonneg.append(nonn)
            else:
                neg.append(S.NegativeOne)
        else:
            if neg:
                nonn = -neg.pop()
                if nonn is not S.One:
                    nonneg.append(nonn)
            else:
                neg.append(S.NegativeOne)
            other.append(I)
        del imag
    if force or e.is_integer:
        cargs = nonneg + neg + other
        other = nc
    else:
        assert not e.is_Integer
        if len(neg) > 1:
            o = S.One
            if not other and neg[0].is_Number:
                o *= neg.pop(0)
            if len(neg) % 2:
                o = -o
            for n in neg:
                nonneg.append(-n)
            if o is not S.One:
                other.append(o)
        elif neg and other:
            if neg[0].is_Number and neg[0] is not S.NegativeOne:
                other.append(S.NegativeOne)
                nonneg.append(-neg[0])
            else:
                other.extend(neg)
        else:
            other.extend(neg)
        del neg
        cargs = nonneg
        other += nc
    rv = S.One
    if cargs:
        if e.is_Rational:
            npow, cargs = sift(cargs, lambda x: x.is_Pow and x.exp.is_Rational and x.base.is_number, binary=True)
            rv = Mul(*[self.func(b.func(*b.args), e) for b in npow])
        rv *= Mul(*[self.func(b, e, evaluate=False) for b in cargs])
    if other:
        rv *= self.func(Mul(*other), e, evaluate=False)
    return rv