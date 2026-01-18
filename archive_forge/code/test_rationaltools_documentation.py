from sympy.polys.rationaltools import together
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import Integral
from sympy.abc import x, y, z
Tests for tools for manipulation of rational expressions. 