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
def _print_Not(self, e):
    from sympy.logic.boolalg import Equivalent, Implies
    if self._use_unicode:
        arg = e.args[0]
        pform = self._print(arg)
        if isinstance(arg, Equivalent):
            return self._print_Equivalent(arg, altchar='⇎')
        if isinstance(arg, Implies):
            return self._print_Implies(arg, altchar='↛')
        if arg.is_Boolean and (not arg.is_Not):
            pform = prettyForm(*pform.parens())
        return prettyForm(*pform.left('¬'))
    else:
        return self._print_Function(e)