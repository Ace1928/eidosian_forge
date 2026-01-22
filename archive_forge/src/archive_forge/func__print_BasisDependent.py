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
def _print_BasisDependent(self, expr):
    from sympy.vector import Vector
    if not self._use_unicode:
        raise NotImplementedError('ASCII pretty printing of BasisDependent is not implemented')
    if expr == expr.zero:
        return prettyForm(expr.zero._pretty_form)
    o1 = []
    vectstrs = []
    if isinstance(expr, Vector):
        items = expr.separate().items()
    else:
        items = [(0, expr)]
    for system, vect in items:
        inneritems = list(vect.components.items())
        inneritems.sort(key=lambda x: x[0].__str__())
        for k, v in inneritems:
            if v == 1:
                o1.append('' + k._pretty_form)
            elif v == -1:
                o1.append('(-1) ' + k._pretty_form)
            else:
                arg_str = self._print(v).parens()[0]
                o1.append(arg_str + ' ' + k._pretty_form)
            vectstrs.append(k._pretty_form)
    if o1[0].startswith(' + '):
        o1[0] = o1[0][3:]
    elif o1[0].startswith(' '):
        o1[0] = o1[0][1:]
    lengths = []
    strs = ['']
    flag = []
    for i, partstr in enumerate(o1):
        flag.append(0)
        if '\n' in partstr:
            tempstr = partstr
            tempstr = tempstr.replace(vectstrs[i], '')
            if '⎟' in tempstr:
                for paren in range(len(tempstr)):
                    flag[i] = 1
                    if tempstr[paren] == '⎟' and tempstr[paren + 1] == '\n':
                        tempstr = tempstr[:paren] + '⎟' + ' ' + vectstrs[i] + tempstr[paren + 1:]
                        break
            elif '⎠' in tempstr:
                index = tempstr.rfind('⎠')
                if index != -1:
                    flag[i] = 1
                    tempstr = tempstr[:index] + '⎠' + ' ' + vectstrs[i] + tempstr[index + 1:]
            o1[i] = tempstr
    o1 = [x.split('\n') for x in o1]
    n_newlines = max([len(x) for x in o1])
    if 1 in flag:
        for i, parts in enumerate(o1):
            if len(parts) == 1:
                parts.insert(0, ' ' * len(parts[0]))
                flag[i] = 1
    for i, parts in enumerate(o1):
        lengths.append(len(parts[flag[i]]))
        for j in range(n_newlines):
            if j + 1 <= len(parts):
                if j >= len(strs):
                    strs.append(' ' * (sum(lengths[:-1]) + 3 * (len(lengths) - 1)))
                if j == flag[i]:
                    strs[flag[i]] += parts[flag[i]] + ' + '
                else:
                    strs[j] += parts[j] + ' ' * (lengths[-1] - len(parts[j]) + 3)
            else:
                if j >= len(strs):
                    strs.append(' ' * (sum(lengths[:-1]) + 3 * (len(lengths) - 1)))
                strs[j] += ' ' * (lengths[-1] + 3)
    return prettyForm('\n'.join([s[:-3] for s in strs]))