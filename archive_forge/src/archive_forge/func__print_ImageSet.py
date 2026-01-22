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
def _print_ImageSet(self, ts):
    if self._use_unicode:
        inn = '∊'
    else:
        inn = 'in'
    fun = ts.lamda
    sets = ts.base_sets
    signature = fun.signature
    expr = self._print(fun.expr)
    if len(signature) == 1:
        S = self._print_seq((signature[0], inn, sets[0]), delimiter=' ')
        return self._hprint_vseparator(expr, S, left='{', right='}', ifascii_nougly=True, delimiter=' ')
    else:
        pargs = tuple((j for var, setv in zip(signature, sets) for j in (var, ' ', inn, ' ', setv, ', ')))
        S = self._print_seq(pargs[:-1], delimiter='')
        return self._hprint_vseparator(expr, S, left='{', right='}', ifascii_nougly=True, delimiter=' ')