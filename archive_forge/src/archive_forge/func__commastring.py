import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
def _commastring(astr):
    startindex = 0
    result = []
    while startindex < len(astr):
        mo = format_re.match(astr, pos=startindex)
        try:
            order1, repeats, order2, dtype = mo.groups()
        except (TypeError, AttributeError):
            raise ValueError(f'format number {len(result) + 1} of "{astr}" is not recognized') from None
        startindex = mo.end()
        if startindex < len(astr):
            if space_re.match(astr, pos=startindex):
                startindex = len(astr)
            else:
                mo = sep_re.match(astr, pos=startindex)
                if not mo:
                    raise ValueError('format number %d of "%s" is not recognized' % (len(result) + 1, astr))
                startindex = mo.end()
        if order2 == '':
            order = order1
        elif order1 == '':
            order = order2
        else:
            order1 = _convorder.get(order1, order1)
            order2 = _convorder.get(order2, order2)
            if order1 != order2:
                raise ValueError('inconsistent byte-order specification %s and %s' % (order1, order2))
            order = order1
        if order in ('|', '=', _nbo):
            order = ''
        dtype = order + dtype
        if repeats == '':
            newitem = dtype
        else:
            newitem = (dtype, ast.literal_eval(repeats))
        result.append(newitem)
    return result