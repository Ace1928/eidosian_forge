import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
@signature({'types': []})
def _func_to_number(self, arg):
    if isinstance(arg, (list, dict, bool)):
        return None
    elif arg is None:
        return None
    elif isinstance(arg, (int, float)):
        return arg
    else:
        try:
            return int(arg)
        except ValueError:
            try:
                return float(arg)
            except ValueError:
                return None