import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
@signature({'types': []})
def _func_type(self, arg):
    if isinstance(arg, STRING_TYPE):
        return 'string'
    elif isinstance(arg, bool):
        return 'boolean'
    elif isinstance(arg, list):
        return 'array'
    elif isinstance(arg, dict):
        return 'object'
    elif isinstance(arg, (float, int)):
        return 'number'
    elif arg is None:
        return 'null'