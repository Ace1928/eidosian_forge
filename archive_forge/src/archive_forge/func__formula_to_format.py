from collections import defaultdict
import re
from .pyutil import memoize
from .periodic import symbols
def _formula_to_format(sub, sup, formula, prefixes=None, infixes=None, suffixes=('(s)', '(l)', '(g)', '(aq)')):
    parts = _formula_to_parts(formula, prefixes.keys(), suffixes)
    stoichs = parts[0].split('..')
    string = ''
    for idx, stoich in enumerate(stoichs):
        if idx == 0:
            m = 1
        else:
            m, stoich = _get_leading_integer(stoich)
            string += _subs('..', infixes)
        if m != 1:
            string += str(m)
        string += re.sub('([0-9]+\\.[0-9]+|[0-9]+)', lambda m: sub(m.group(1)), stoich)
    if parts[1] is not None:
        chg = _get_charge(parts[1])
        if chg < 0:
            token = '-' if chg == -1 else '%d-' % -chg
        if chg > 0:
            token = '+' if chg == 1 else '%d+' % chg
        string += sup(token)
    if len(parts) > 4:
        raise ValueError('Incorrect formula')
    pre_str = ''.join(map(lambda x: _subs(x, prefixes), parts[2]))
    return pre_str + string + ''.join(parts[3])