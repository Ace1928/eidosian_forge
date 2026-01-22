from itertools import product
from typing import Tuple as tTuple
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import (Function, ArgumentIndexError, expand_log,
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational, pi, I, ImaginaryUnit
from sympy.core.parameters import global_parameters
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import arg, unpolarify, im, re, Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory import multiplicity, perfect_power
from sympy.ntheory.factor_ import factorint
@cacheit
def _log_atan_table():
    return {sqrt(3): pi / 3, 1: pi / 4, sqrt(5 - 2 * sqrt(5)): pi / 5, sqrt(2) * sqrt(5 - sqrt(5)) / (1 + sqrt(5)): pi / 5, sqrt(5 + 2 * sqrt(5)): pi * Rational(2, 5), sqrt(2) * sqrt(sqrt(5) + 5) / (-1 + sqrt(5)): pi * Rational(2, 5), sqrt(3) / 3: pi / 6, sqrt(2) - 1: pi / 8, sqrt(2 - sqrt(2)) / sqrt(sqrt(2) + 2): pi / 8, sqrt(2) + 1: pi * Rational(3, 8), sqrt(sqrt(2) + 2) / sqrt(2 - sqrt(2)): pi * Rational(3, 8), sqrt(1 - 2 * sqrt(5) / 5): pi / 10, (-sqrt(2) + sqrt(10)) / (2 * sqrt(sqrt(5) + 5)): pi / 10, sqrt(1 + 2 * sqrt(5) / 5): pi * Rational(3, 10), (sqrt(2) + sqrt(10)) / (2 * sqrt(5 - sqrt(5))): pi * Rational(3, 10), 2 - sqrt(3): pi / 12, (-1 + sqrt(3)) / (1 + sqrt(3)): pi / 12, 2 + sqrt(3): pi * Rational(5, 12), (1 + sqrt(3)) / (-1 + sqrt(3)): pi * Rational(5, 12)}