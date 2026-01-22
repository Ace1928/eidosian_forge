from __future__ import annotations
from typing import Any
from functools import wraps
from itertools import chain
from sympy.core import S
from sympy.core.numbers import equal_valued
from sympy.codegen.ast import (
from sympy.printing.codeprinter import CodePrinter, requires
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range
from sympy.printing.codeprinter import ccode, print_ccode # noqa:F401
def _print_Label(self, expr):
    if expr.body == none:
        return '%s:' % str(expr.name)
    if len(expr.body.args) == 1:
        return '%s:\n%s' % (str(expr.name), self._print_CodeBlock(expr.body))
    return '%s:\n{\n%s\n}' % (str(expr.name), self._print_CodeBlock(expr.body))