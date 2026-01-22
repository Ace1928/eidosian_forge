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
def _print_Sum(self, expr):
    ascii_mode = not self._use_unicode

    def asum(hrequired, lower, upper, use_ascii):

        def adjust(s, wid=None, how='<^>'):
            if not wid or len(s) > wid:
                return s
            need = wid - len(s)
            if how in ('<^>', '<') or how not in list('<^>'):
                return s + ' ' * need
            half = need // 2
            lead = ' ' * half
            if how == '>':
                return ' ' * need + s
            return lead + s + ' ' * (need - len(lead))
        h = max(hrequired, 2)
        d = h // 2
        w = d + 1
        more = hrequired % 2
        lines = []
        if use_ascii:
            lines.append('_' * w + ' ')
            lines.append('\\%s`' % (' ' * (w - 1)))
            for i in range(1, d):
                lines.append('%s\\%s' % (' ' * i, ' ' * (w - i)))
            if more:
                lines.append('%s)%s' % (' ' * d, ' ' * (w - d)))
            for i in reversed(range(1, d)):
                lines.append('%s/%s' % (' ' * i, ' ' * (w - i)))
            lines.append('/' + '_' * (w - 1) + ',')
            return (d, h + more, lines, more)
        else:
            w = w + more
            d = d + more
            vsum = vobj('sum', 4)
            lines.append('_' * w)
            for i in range(0, d):
                lines.append('%s%s%s' % (' ' * i, vsum[2], ' ' * (w - i - 1)))
            for i in reversed(range(0, d)):
                lines.append('%s%s%s' % (' ' * i, vsum[4], ' ' * (w - i - 1)))
            lines.append(vsum[8] * w)
            return (d, h + 2 * more, lines, more)
    f = expr.function
    prettyF = self._print(f)
    if f.is_Add:
        prettyF = prettyForm(*prettyF.parens())
    H = prettyF.height() + 2
    first = True
    max_upper = 0
    sign_height = 0
    for lim in expr.limits:
        prettyLower, prettyUpper = self.__print_SumProduct_Limits(lim)
        max_upper = max(max_upper, prettyUpper.height())
        d, h, slines, adjustment = asum(H, prettyLower.width(), prettyUpper.width(), ascii_mode)
        prettySign = stringPict('')
        prettySign = prettyForm(*prettySign.stack(*slines))
        if first:
            sign_height = prettySign.height()
        prettySign = prettyForm(*prettySign.above(prettyUpper))
        prettySign = prettyForm(*prettySign.below(prettyLower))
        if first:
            prettyF.baseline -= d - (prettyF.height() // 2 - prettyF.baseline)
            first = False
        pad = stringPict('')
        pad = prettyForm(*pad.stack(*[' '] * h))
        prettySign = prettyForm(*prettySign.right(pad))
        prettyF = prettyForm(*prettySign.right(prettyF))
    ascii_adjustment = ascii_mode if not adjustment else 0
    prettyF.baseline = max_upper + sign_height // 2 + ascii_adjustment
    prettyF.binding = prettyForm.MUL
    return prettyF