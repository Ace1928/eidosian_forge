from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _module_format(self, fqn, register=True):
    parts = fqn.split('.')
    if register and len(parts) > 1:
        self.module_imports['.'.join(parts[:-1])].add(parts[-1])
    if self._settings['fully_qualified_modules']:
        return fqn
    else:
        return fqn.split('(')[0].split('[')[0].split('.')[-1]