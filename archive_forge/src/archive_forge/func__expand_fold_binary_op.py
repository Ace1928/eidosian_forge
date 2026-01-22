from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _expand_fold_binary_op(self, op, args):
    """
        This method expands a fold on binary operations.

        ``functools.reduce`` is an example of a folded operation.

        For example, the expression

        `A + B + C + D`

        is folded into

        `((A + B) + C) + D`
        """
    if len(args) == 1:
        return self._print(args[0])
    else:
        return '%s(%s, %s)' % (self._module_format(op), self._expand_fold_binary_op(op, args[:-1]), self._print(args[-1]))