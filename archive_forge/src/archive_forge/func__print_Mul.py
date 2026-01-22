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
def _print_Mul(self, product):
    from sympy.physics.units import Quantity
    args = product.args
    if args[0] is S.One or any((isinstance(arg, Number) for arg in args[1:])):
        strargs = list(map(self._print, args))
        negone = strargs[0] == '-1'
        if negone:
            strargs[0] = prettyForm('1', 0, 0)
        obj = prettyForm.__mul__(*strargs)
        if negone:
            obj = prettyForm('-' + obj.s, obj.baseline, obj.binding)
        return obj
    a = []
    b = []
    if self.order not in ('old', 'none'):
        args = product.as_ordered_factors()
    else:
        args = list(product.args)
    args = sorted(args, key=lambda x: isinstance(x, Quantity) or (isinstance(x, Pow) and isinstance(x.base, Quantity)))
    for item in args:
        if item.is_commutative and item.is_Pow and item.exp.is_Rational and item.exp.is_negative:
            if item.exp != -1:
                b.append(Pow(item.base, -item.exp, evaluate=False))
            else:
                b.append(Pow(item.base, -item.exp))
        elif item.is_Rational and item is not S.Infinity:
            if item.p != 1:
                a.append(Rational(item.p))
            if item.q != 1:
                b.append(Rational(item.q))
        else:
            a.append(item)
    a = [self._print(ai) for ai in a]
    b = [self._print(bi) for bi in b]
    if len(b) == 0:
        return prettyForm.__mul__(*a)
    else:
        if len(a) == 0:
            a.append(self._print(S.One))
        return prettyForm.__mul__(*a) / prettyForm.__mul__(*b)