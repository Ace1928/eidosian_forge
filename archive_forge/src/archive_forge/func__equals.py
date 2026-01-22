import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def _equals(x, y):
    if _is_special_integer_case(x, y):
        return False
    else:
        return x == y