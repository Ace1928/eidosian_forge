import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def _is_comparable(x):
    return _is_actual_number(x) or isinstance(x, string_type)