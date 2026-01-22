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
def _print_Type(self, type_):
    type_ = self.type_aliases.get(type_, type_)
    type_str = self.type_mappings.get(type_, type_.name)
    module_uses = self.type_modules.get(type_)
    if module_uses:
        for k, v in module_uses:
            self.module_uses[k].add(v)
    return type_str