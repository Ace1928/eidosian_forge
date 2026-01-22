from collections import defaultdict
import re
from .pyutil import memoize
from .periodic import symbols
def _parse_stoich(stoich):
    if stoich == 'e':
        return {}
    comp = {}
    for k, n in _get_formula_parser().parseString(stoich, parseAll=True):
        if n == int(n):
            comp[symbols.index(k) + 1] = int(n)
        else:
            comp[symbols.index(k) + 1] = n
    return comp