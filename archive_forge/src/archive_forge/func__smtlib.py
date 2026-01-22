import contextlib
import itertools
import re
import typing
from enum import Enum
from typing import Callable
import sympy
from sympy import Add, Implies, sqrt
from sympy.core import Mul, Pow
from sympy.core import (S, pi, symbols, Function, Rational, Integer,
from sympy.functions import Piecewise, exp, sin, cos
from sympy.printing.smtlib import smtlib_code
from sympy.testing.pytest import raises, Failed
def _smtlib(self, printer):
    bound_symbol_declarations = [printer._s_expr(sym.name, [printer._known_types[printer.symbol_table[sym]], Interval(start, end)]) for sym, start, end in self.limits]
    return printer._s_expr('forall', [printer._s_expr('', bound_symbol_declarations), self.function])