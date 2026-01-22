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
class HyperRep_power1(HyperRep):
    """ Return a representative for hyper([-a], [], z) == (1 - z)**a. """

    @classmethod
    def _expr_small(cls, a, x):
        return (1 - x) ** a

    @classmethod
    def _expr_small_minus(cls, a, x):
        return (1 + x) ** a

    @classmethod
    def _expr_big(cls, a, x, n):
        if a.is_integer:
            return cls._expr_small(a, x)
        return (x - 1) ** a * exp((2 * n - 1) * pi * I * a)

    @classmethod
    def _expr_big_minus(cls, a, x, n):
        if a.is_integer:
            return cls._expr_small_minus(a, x)
        return (1 + x) ** a * exp(2 * n * pi * I * a)