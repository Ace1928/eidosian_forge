from typing import Tuple as tTuple
from sympy.core import S, Add, Mul, sympify, Symbol, Dummy, Basic
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import (Function, Derivative, ArgumentIndexError,
from sympy.core.logic import fuzzy_not, fuzzy_or
from sympy.core.numbers import pi, I, oo
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
@classmethod
def _getunbranched(cls, ar):
    from sympy.functions.elementary.exponential import exp_polar, log
    if ar.is_Mul:
        args = ar.args
    else:
        args = [ar]
    unbranched = 0
    for a in args:
        if not a.is_polar:
            unbranched += arg(a)
        elif isinstance(a, exp_polar):
            unbranched += a.exp.as_real_imag()[1]
        elif a.is_Pow:
            re, im = a.exp.as_real_imag()
            unbranched += re * unbranched_argument(a.base) + im * log(abs(a.base))
        elif isinstance(a, polar_lift):
            unbranched += arg(a.args[0])
        else:
            return None
    return unbranched