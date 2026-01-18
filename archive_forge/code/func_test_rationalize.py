import sys
import builtins
import types
from sympy.assumptions import Q
from sympy.core import Symbol, Function, Float, Rational, Integer, I, Mul, Pow, Eq, Lt, Le, Gt, Ge, Ne
from sympy.functions import exp, factorial, factorial2, sin, Min, Max
from sympy.logic import And
from sympy.series import Limit
from sympy.testing.pytest import raises, skip
from sympy.parsing.sympy_parser import (
def test_rationalize():
    inputs = {'0.123': Rational(123, 1000)}
    transformations = standard_transformations + (rationalize,)
    for text, result in inputs.items():
        assert parse_expr(text, transformations=transformations) == result