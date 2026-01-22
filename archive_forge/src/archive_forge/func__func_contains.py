import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
@signature({'types': ['array', 'string']}, {'types': []})
def _func_contains(self, subject, search):
    return search in subject