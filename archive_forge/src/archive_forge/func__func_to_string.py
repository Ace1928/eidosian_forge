import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
@signature({'types': []})
def _func_to_string(self, arg):
    if isinstance(arg, STRING_TYPE):
        return arg
    else:
        return json.dumps(arg, separators=(',', ':'), default=str)