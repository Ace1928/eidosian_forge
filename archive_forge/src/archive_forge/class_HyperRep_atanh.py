from functools import reduce
from sympy.core import S, ilcm, Mod
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Function, Derivative, ArgumentIndexError
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.numbers import I, pi, oo, zoo
from sympy.core.relational import Ne
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.functions import (sqrt, exp, log, sin, cos, asin, atan,
from sympy.functions import factorial, RisingFactorial
from sympy.functions.elementary.complexes import Abs, re, unpolarify
from sympy.functions.elementary.exponential import exp_polar
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import (And, Or)
class HyperRep_atanh(HyperRep):
    """ Represent hyper([1/2, 1], [3/2], z) == atanh(sqrt(z))/sqrt(z). """

    @classmethod
    def _expr_small(cls, x):
        return atanh(sqrt(x)) / sqrt(x)

    def _expr_small_minus(cls, x):
        return atan(sqrt(x)) / sqrt(x)

    def _expr_big(cls, x, n):
        if n.is_even:
            return (acoth(sqrt(x)) + I * pi / 2) / sqrt(x)
        else:
            return (acoth(sqrt(x)) - I * pi / 2) / sqrt(x)

    def _expr_big_minus(cls, x, n):
        if n.is_even:
            return atan(sqrt(x)) / sqrt(x)
        else:
            return (atan(sqrt(x)) - pi) / sqrt(x)