import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def _is_special_number_case(x, y):
    if _is_actual_number(x) and x in (0, 1):
        return isinstance(y, bool)
    elif _is_actual_number(y) and y in (0, 1):
        return isinstance(x, bool)