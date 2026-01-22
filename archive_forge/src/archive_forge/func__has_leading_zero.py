import functools
import re
import warnings
def _has_leading_zero(value):
    return value and value[0] == '0' and value.isdigit() and (value != '0')