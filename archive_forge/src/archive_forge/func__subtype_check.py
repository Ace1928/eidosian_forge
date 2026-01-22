import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
def _subtype_check(self, current, allowed_subtypes, types, function_name):
    if len(allowed_subtypes) == 1:
        allowed_subtypes = allowed_subtypes[0]
        for element in current:
            actual_typename = type(element).__name__
            if actual_typename not in allowed_subtypes:
                raise exceptions.JMESPathTypeError(function_name, element, actual_typename, types)
    elif len(allowed_subtypes) > 1 and current:
        first = type(current[0]).__name__
        for subtypes in allowed_subtypes:
            if first in subtypes:
                allowed = subtypes
                break
        else:
            raise exceptions.JMESPathTypeError(function_name, current[0], first, types)
        for element in current:
            actual_typename = type(element).__name__
            if actual_typename not in allowed:
                raise exceptions.JMESPathTypeError(function_name, element, actual_typename, types)