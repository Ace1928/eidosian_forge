from collections import OrderedDict, deque
from datetime import date, time, datetime
from decimal import Decimal
from fractions import Fraction
import ast
import enum
import typing
def _is_standard_types_deep(x, check_dict_values: bool, deep: bool):
    typ = type(x)
    if is_any(typ, str, int, bool, float, bytes, complex, date, time, datetime, Fraction, Decimal, type(None), object):
        return (True, 0)
    if is_any(typ, tuple, frozenset, list, set, dict, OrderedDict, deque, slice):
        if typ in [slice]:
            length = 0
        else:
            length = len(x)
        assert isinstance(deep, bool)
        if not deep:
            return (True, length)
        if check_dict_values and typ in (dict, OrderedDict):
            items = (v for pair in x.items() for v in pair)
        elif typ is slice:
            items = [x.start, x.stop, x.step]
        else:
            items = x
        for item in items:
            if length > 100000:
                return (False, length)
            is_standard, item_length = _is_standard_types_deep(item, check_dict_values, deep)
            if not is_standard:
                return (False, length)
            length += item_length
        return (True, length)
    return (False, 0)