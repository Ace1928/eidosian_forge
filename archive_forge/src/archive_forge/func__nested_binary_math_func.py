from __future__ import annotations
from typing import Any
from sympy.core import Mul, Pow, S, Rational
from sympy.core.mul import _keep_coeff
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from re import search
def _nested_binary_math_func(self, expr):
    return '{name}({arg1}, {arg2})'.format(name=self.known_functions[expr.__class__.__name__], arg1=self._print(expr.args[0]), arg2=self._print(expr.func(*expr.args[1:])))