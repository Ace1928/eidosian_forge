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
def _print_Range(self, s):
    if self._use_unicode:
        dots = '…'
    else:
        dots = '...'
    if s.start.is_infinite and s.stop.is_infinite:
        if s.step.is_positive:
            printset = (dots, -1, 0, 1, dots)
        else:
            printset = (dots, 1, 0, -1, dots)
    elif s.start.is_infinite:
        printset = (dots, s[-1] - s.step, s[-1])
    elif s.stop.is_infinite:
        it = iter(s)
        printset = (next(it), next(it), dots)
    elif len(s) > 4:
        it = iter(s)
        printset = (next(it), next(it), dots, s[-1])
    else:
        printset = tuple(s)
    return self._print_seq(printset, '{', '}', ', ')