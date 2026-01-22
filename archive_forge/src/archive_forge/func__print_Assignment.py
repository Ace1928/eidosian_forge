from __future__ import annotations
from typing import Any
from sympy.core import S, Rational, Float, Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
def _print_Assignment(self, expr):
    from sympy.tensor.indexed import IndexedBase
    lhs = expr.lhs
    rhs = expr.rhs
    if self._settings['contract'] and (lhs.has(IndexedBase) or rhs.has(IndexedBase)):
        return self._doprint_loops(rhs, lhs)
    else:
        lhs_code = self._print(lhs)
        rhs_code = self._print(rhs)
        return self._get_statement('%s = %s' % (lhs_code, rhs_code))