import re
import operator
from fractions import Fraction
import sys
def _combine_dicts(list_of_dicts, combine_function):
    """
    >>> d = _combine_dicts(
    ...      [ {'key1': 1, 'key2': 2},
    ...        {'key1': 1} ],
    ...      combine_function = operator.add)
    >>> d['key1']
    2
    >>> d['key2']
    2
    """
    result = {}
    for a_dict in list_of_dicts:
        for k, v in list(a_dict.items()):
            if k in result:
                result[k] = combine_function(result[k], v)
            else:
                result[k] = v
    return result