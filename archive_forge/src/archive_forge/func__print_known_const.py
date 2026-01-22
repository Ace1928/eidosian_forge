from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _print_known_const(self, expr):
    known = self.known_constants[expr.__class__.__name__]
    return self._module_format(known)