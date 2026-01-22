import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
@signature({'types': ['array']}, {'types': ['expref']})
def _func_min_by(self, array, expref):
    keyfunc = self._create_key_func(expref, ['number', 'string'], 'min_by')
    if array:
        return min(array, key=keyfunc)
    else:
        return None