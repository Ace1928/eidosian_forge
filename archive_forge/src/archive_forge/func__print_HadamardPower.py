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
def _print_HadamardPower(self, expr):
    if self._use_unicode:
        circ = pretty_atom('Ring')
    else:
        circ = self._print('.')
    pretty_base = self._print(expr.base)
    pretty_exp = self._print(expr.exp)
    if precedence(expr.exp) < PRECEDENCE['Mul']:
        pretty_exp = prettyForm(*pretty_exp.parens())
    pretty_circ_exp = prettyForm(*stringPict.next(circ, pretty_exp), binding=prettyForm.LINE)
    return pretty_base ** pretty_circ_exp