from __future__ import annotations
from typing import Any
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range
def _print_AugmentedAssignment(self, expr):
    lhs_code = self._print(expr.lhs)
    op = expr.op
    rhs_code = self._print(expr.rhs)
    return '{} {} {};'.format(lhs_code, op, rhs_code)