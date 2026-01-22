import functools, itertools
from sympy.core.sympify import _sympify, sympify
from sympy.core.expr import Expr
from sympy.core import Basic, Tuple
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.core.symbol import Symbol
from sympy.core.numbers import Integer
@classmethod
def _calculate_shape_from_limits(cls, limits):
    return tuple([sup - inf + 1 for _, inf, sup in limits])