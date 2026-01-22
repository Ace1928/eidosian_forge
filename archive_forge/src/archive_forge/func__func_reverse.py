import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
@signature({'types': ['array', 'string']})
def _func_reverse(self, arg):
    if isinstance(arg, STRING_TYPE):
        return arg[::-1]
    else:
        return list(reversed(arg))