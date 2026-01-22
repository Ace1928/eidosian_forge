from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _print_Stream(self, strm):
    if str(strm.name) == 'stdout':
        return self._module_format('sys.stdout')
    elif str(strm.name) == 'stderr':
        return self._module_format('sys.stderr')
    else:
        return self._print(strm.name)