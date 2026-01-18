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
def test_split_symbols():
    transformations = standard_transformations + (split_symbols, implicit_multiplication)
    x = Symbol('x')
    y = Symbol('y')
    xy = Symbol('xy')
    assert parse_expr('xy') == xy
    assert parse_expr('xy', transformations=transformations) == x * y