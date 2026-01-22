import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
@signature({'types': [], 'variadic': True})
def _func_not_null(self, *arguments):
    for argument in arguments:
        if argument is not None:
            return argument