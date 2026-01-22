import functools, itertools
from sympy.core.sympify import _sympify, sympify
from sympy.core.expr import Expr
from sympy.core import Basic, Tuple
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.core.symbol import Symbol
from sympy.core.numbers import Integer
@classmethod
def _calculate_loop_size(cls, shape):
    if not shape:
        return 0
    loop_size = 1
    for l in shape:
        loop_size = loop_size * l
    return loop_size