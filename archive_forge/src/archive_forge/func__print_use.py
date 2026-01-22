from __future__ import annotations
from typing import Any
from collections import defaultdict
from itertools import chain
import string
from sympy.codegen.ast import (
from sympy.codegen.fnodes import (
from sympy.core import S, Add, N, Float, Symbol
from sympy.core.function import Function
from sympy.core.numbers import equal_valued
from sympy.core.relational import Eq
from sympy.sets import Range
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.printing.printer import printer_context
from sympy.printing.codeprinter import fcode, print_fcode # noqa:F401
def _print_use(self, use):
    result = 'use %s' % self._print(use.namespace)
    if use.rename != None:
        result += ', ' + ', '.join([self._print(rnm) for rnm in use.rename])
    if use.only != None:
        result += ', only: ' + ', '.join([self._print(nly) for nly in use.only])
    return result