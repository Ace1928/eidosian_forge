from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _print_PermuteDims(self, expr):
    return '%s(%s, %s)' % (self._module_format(self._module + '.' + self._transpose), self._print(expr.expr), self._print(expr.permutation.array_form))