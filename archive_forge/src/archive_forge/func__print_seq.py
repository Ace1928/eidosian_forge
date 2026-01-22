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
def _print_seq(self, seq, left=None, right=None, delimiter=', ', parenthesize=lambda x: False, ifascii_nougly=True):
    try:
        pforms = []
        for item in seq:
            pform = self._print(item)
            if parenthesize(item):
                pform = prettyForm(*pform.parens())
            if pforms:
                pforms.append(delimiter)
            pforms.append(pform)
        if not pforms:
            s = stringPict('')
        else:
            s = prettyForm(*stringPict.next(*pforms))
    except AttributeError:
        s = None
        for item in seq:
            pform = self.doprint(item)
            if parenthesize(item):
                pform = prettyForm(*pform.parens())
            if s is None:
                s = pform
            else:
                s = prettyForm(*stringPict.next(s, delimiter))
                s = prettyForm(*stringPict.next(s, pform))
        if s is None:
            s = stringPict('')
    s = prettyForm(*s.parens(left, right, ifascii_nougly=ifascii_nougly))
    return s