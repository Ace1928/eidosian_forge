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
def _eval_rewrite_as_Piecewise(self, arg, **kwargs):
    if arg.is_extended_real:
        return Piecewise((arg, arg >= 0), (-arg, True))
    elif arg.is_imaginary:
        return Piecewise((I * arg, I * arg >= 0), (-I * arg, True))