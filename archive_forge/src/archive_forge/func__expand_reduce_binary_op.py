from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _expand_reduce_binary_op(self, op, args):
    """
        This method expands a reductin on binary operations.

        Notice: this is NOT the same as ``functools.reduce``.

        For example, the expression

        `A + B + C + D`

        is reduced into:

        `(A + B) + (C + D)`
        """
    if len(args) == 1:
        return self._print(args[0])
    else:
        N = len(args)
        Nhalf = N // 2
        return '%s(%s, %s)' % (self._module_format(op), self._expand_reduce_binary_op(args[:Nhalf]), self._expand_reduce_binary_op(args[Nhalf:]))