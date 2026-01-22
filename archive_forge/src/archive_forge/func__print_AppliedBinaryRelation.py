from __future__ import annotations
from typing import Any
from sympy.core import S, Rational, Pow, Basic, Mul, Number
from sympy.core.mul import _keep_coeff
from sympy.core.relational import Relational
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import SympifyError
from sympy.utilities.iterables import sift
from .precedence import precedence, PRECEDENCE
from .printer import Printer, print_function
from mpmath.libmp import prec_to_dps, to_str as mlib_to_str
def _print_AppliedBinaryRelation(self, expr):
    rel = expr.function
    return '%s(%s, %s)' % (self._print(rel), self._print(expr.lhs), self._print(expr.rhs))