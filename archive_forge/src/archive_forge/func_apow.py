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
def apow(i):
    b, e = i.as_base_exp()
    eargs = list(Mul.make_args(e))
    if eargs[0] is S.NegativeOne:
        eargs = eargs[1:]
    else:
        eargs[0] = -eargs[0]
    e = Mul._from_args(eargs)
    if isinstance(i, Pow):
        return i.func(b, e, evaluate=False)
    return i.func(e, evaluate=False)