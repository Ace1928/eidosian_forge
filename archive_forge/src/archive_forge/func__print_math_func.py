from __future__ import annotations
from typing import Any
from functools import wraps
from itertools import chain
from sympy.core import S
from sympy.core.numbers import equal_valued
from sympy.codegen.ast import (
from sympy.printing.codeprinter import CodePrinter, requires
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range
from sympy.printing.codeprinter import ccode, print_ccode # noqa:F401
@requires(headers={'math.h'}, libraries={'m'})
@_as_macro_if_defined
def _print_math_func(self, expr, nest=False, known=None):
    if known is None:
        known = self.known_functions[expr.__class__.__name__]
    if not isinstance(known, str):
        for cb, name in known:
            if cb(*expr.args):
                known = name
                break
        else:
            raise ValueError('No matching printer')
    try:
        return known(self, *expr.args)
    except TypeError:
        suffix = self._get_func_suffix(real) if self._ns + known in self._prec_funcs else ''
    if nest:
        args = self._print(expr.args[0])
        if len(expr.args) > 1:
            paren_pile = ''
            for curr_arg in expr.args[1:-1]:
                paren_pile += ')'
                args += ', {ns}{name}{suffix}({next}'.format(ns=self._ns, name=known, suffix=suffix, next=self._print(curr_arg))
            args += ', %s%s' % (self._print(expr.func(expr.args[-1])), paren_pile)
    else:
        args = ', '.join((self._print(arg) for arg in expr.args))
    return '{ns}{name}{suffix}({args})'.format(ns=self._ns, name=known, suffix=suffix, args=args)