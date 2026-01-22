from collections import defaultdict
import re
from .pyutil import memoize
from .periodic import symbols
def _formula_to_parts(formula, prefixes, suffixes):
    drop_pref, drop_suff = ([], [])
    for ign in prefixes:
        if formula.startswith(ign):
            drop_pref.append(ign)
            formula = formula[len(ign):]
    for ign in suffixes:
        if formula.endswith(ign):
            drop_suff.append(ign)
            formula = formula[:-len(ign)]
    if '/' in formula:
        raise ValueError("Slashes ('/') in charge strings are deprecated.  Use `Fe+3` instead of `Fe/3+`.")
    else:
        for token in '+-':
            if token in formula:
                if formula.count(token) > 1:
                    raise ValueError('Multiple tokens: %s' % token)
                parts = formula.split(token)
                parts[1] = token + parts[1]
                break
        else:
            parts = [formula, None]
    return parts + [tuple(drop_pref), tuple(drop_suff[::-1])]