from sympy.core.function import (Derivative, Function, diff)
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import sin
from sympy.core.multidimensional import vectorize
@vectorize(0, 1)
def vdiff(f, y):
    return diff(f, y)