import itertools
from sympy.core import S
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import Number, Rational
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.core.sympify import SympifyError
from sympy.printing.conventions import requires_partial
from sympy.printing.precedence import PRECEDENCE, precedence, precedence_traditional
from sympy.printing.printer import Printer, print_function
from sympy.printing.str import sstr
from sympy.utilities.iterables import has_variety
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import hobj, vobj, xobj, \
def _print_Order(self, expr):
    pform = self._print(expr.expr)
    if expr.point and any((p != S.Zero for p in expr.point)) or len(expr.variables) > 1:
        pform = prettyForm(*pform.right('; '))
        if len(expr.variables) > 1:
            pform = prettyForm(*pform.right(self._print(expr.variables)))
        elif len(expr.variables):
            pform = prettyForm(*pform.right(self._print(expr.variables[0])))
        if self._use_unicode:
            pform = prettyForm(*pform.right(' → '))
        else:
            pform = prettyForm(*pform.right(' -> '))
        if len(expr.point) > 1:
            pform = prettyForm(*pform.right(self._print(expr.point)))
        else:
            pform = prettyForm(*pform.right(self._print(expr.point[0])))
    pform = prettyForm(*pform.parens())
    pform = prettyForm(*pform.left('O'))
    return pform