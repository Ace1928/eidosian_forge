from sympy.core import S, sympify, cacheit
from sympy.core.add import Add
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.logic import fuzzy_or, fuzzy_and, FuzzyBool
from sympy.core.numbers import I, pi, Rational
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import (binomial, factorial,
from sympy.functions.combinatorial.numbers import bernoulli, euler, nC
from sympy.functions.elementary.complexes import Abs, im, re
from sympy.functions.elementary.exponential import exp, log, match_real_imag
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (
from sympy.polys.specialpolys import symmetric_poly
@cacheit
def _acsch_table():
    return {I: -pi / 2, I * (sqrt(2) + sqrt(6)): -pi / 12, I * (1 + sqrt(5)): -pi / 10, I * 2 / sqrt(2 - sqrt(2)): -pi / 8, I * 2: -pi / 6, I * sqrt(2 + 2 / sqrt(5)): -pi / 5, I * sqrt(2): -pi / 4, I * (sqrt(5) - 1): -3 * pi / 10, I * 2 / sqrt(3): -pi / 3, I * 2 / sqrt(2 + sqrt(2)): -3 * pi / 8, I * sqrt(2 - 2 / sqrt(5)): -2 * pi / 5, I * (sqrt(6) - sqrt(2)): -5 * pi / 12, S(2): -I * log((1 + sqrt(5)) / 2)}