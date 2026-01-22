from __future__ import annotations
from typing import Any
from sympy.core import Mul, Pow, S, Rational
from sympy.core.mul import _keep_coeff
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from re import search
def _one_or_two_reversed_args(self, expr):
    assert len(expr.args) <= 2
    return '{name}({args})'.format(name=self.known_functions[expr.__class__.__name__], args=', '.join([self._print(x) for x in reversed(expr.args)]))