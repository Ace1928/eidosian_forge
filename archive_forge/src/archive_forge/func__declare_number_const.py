from __future__ import annotations
from typing import Any
from sympy.core import S, Rational, Float, Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
def _declare_number_const(self, name, value):
    return 'const %s: f64 = %s;' % (name, value)