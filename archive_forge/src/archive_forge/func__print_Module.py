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
def _print_Module(self, mod):
    return 'module {name}\n{declarations}\n\ncontains\n\n{definitions}\nend module\n'.format(**mod.kwargs(apply=lambda arg: self._print(arg)))