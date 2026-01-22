from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _print_While(self, whl):
    body = '\n'.join((self._print(arg) for arg in whl.body))
    return 'while {cond}:\n{body}'.format(cond=self._print(whl.condition), body=self._indent_codestring(body))