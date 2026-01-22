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
def _print_Product(self, expr):
    func = expr.term
    pretty_func = self._print(func)
    horizontal_chr = xobj('_', 1)
    corner_chr = xobj('_', 1)
    vertical_chr = xobj('|', 1)
    if self._use_unicode:
        horizontal_chr = xobj('-', 1)
        corner_chr = '┬'
    func_height = pretty_func.height()
    first = True
    max_upper = 0
    sign_height = 0
    for lim in expr.limits:
        pretty_lower, pretty_upper = self.__print_SumProduct_Limits(lim)
        width = (func_height + 2) * 5 // 3 - 2
        sign_lines = [horizontal_chr + corner_chr + horizontal_chr * (width - 2) + corner_chr + horizontal_chr]
        for _ in range(func_height + 1):
            sign_lines.append(' ' + vertical_chr + ' ' * (width - 2) + vertical_chr + ' ')
        pretty_sign = stringPict('')
        pretty_sign = prettyForm(*pretty_sign.stack(*sign_lines))
        max_upper = max(max_upper, pretty_upper.height())
        if first:
            sign_height = pretty_sign.height()
        pretty_sign = prettyForm(*pretty_sign.above(pretty_upper))
        pretty_sign = prettyForm(*pretty_sign.below(pretty_lower))
        if first:
            pretty_func.baseline = 0
            first = False
        height = pretty_sign.height()
        padding = stringPict('')
        padding = prettyForm(*padding.stack(*[' '] * (height - 1)))
        pretty_sign = prettyForm(*pretty_sign.right(padding))
        pretty_func = prettyForm(*pretty_sign.right(pretty_func))
    pretty_func.baseline = max_upper + sign_height // 2
    pretty_func.binding = prettyForm.MUL
    return pretty_func