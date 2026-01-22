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
def _print_Piecewise(self, pexpr):
    P = {}
    for n, ec in enumerate(pexpr.args):
        P[n, 0] = self._print(ec.expr)
        if ec.cond == True:
            P[n, 1] = prettyForm('otherwise')
        else:
            P[n, 1] = prettyForm(*prettyForm('for ').right(self._print(ec.cond)))
    hsep = 2
    vsep = 1
    len_args = len(pexpr.args)
    maxw = [max([P[i, j].width() for i in range(len_args)]) for j in range(2)]
    D = None
    for i in range(len_args):
        D_row = None
        for j in range(2):
            p = P[i, j]
            assert p.width() <= maxw[j]
            wdelta = maxw[j] - p.width()
            wleft = wdelta // 2
            wright = wdelta - wleft
            p = prettyForm(*p.right(' ' * wright))
            p = prettyForm(*p.left(' ' * wleft))
            if D_row is None:
                D_row = p
                continue
            D_row = prettyForm(*D_row.right(' ' * hsep))
            D_row = prettyForm(*D_row.right(p))
        if D is None:
            D = D_row
            continue
        for _ in range(vsep):
            D = prettyForm(*D.below(' '))
        D = prettyForm(*D.below(D_row))
    D = prettyForm(*D.parens('{', ''))
    D.baseline = D.height() // 2
    D.binding = prettyForm.OPEN
    return D