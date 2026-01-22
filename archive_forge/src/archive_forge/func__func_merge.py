import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
@signature({'types': ['object'], 'variadic': True})
def _func_merge(self, *arguments):
    merged = {}
    for arg in arguments:
        merged.update(arg)
    return merged