import collections
import operator
from functools import reduce
from collections.abc import Mapping
def _get_factory(f, kwargs):
    factory = kwargs.pop('factory', dict)
    if kwargs:
        raise TypeError(f"{f.__name__}() got an unexpected keyword argument '{kwargs.popitem()[0]}'")
    return factory