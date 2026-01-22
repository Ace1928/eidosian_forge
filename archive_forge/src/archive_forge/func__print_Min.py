from itertools import chain
from sympy.codegen.ast import Type, none
from .c import C89CodePrinter, C99CodePrinter
from sympy.printing.codeprinter import cxxcode # noqa:F401
def _print_Min(self, expr):
    from sympy.functions.elementary.miscellaneous import Min
    if len(expr.args) == 1:
        return self._print(expr.args[0])
    return '%smin(%s, %s)' % (self._ns, self._print(expr.args[0]), self._print(Min(*expr.args[1:])))