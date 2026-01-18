from sympy.core.numbers import (I, Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.calculus.singularities import (
from sympy.sets import Interval, FiniteSet
from sympy.testing.pytest import raises
from sympy.abc import x, y
Test whether is_monotonic returns correct value.