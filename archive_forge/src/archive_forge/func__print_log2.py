from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _print_log2(self, e):
    return '{0}({1})/{0}(2)'.format(self._module_format('mpmath.log'), self._print(e.args[0]))