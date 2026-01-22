import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
def _type_check_single(self, current, types, function_name):
    allowed_types, allowed_subtypes = self._get_allowed_pytypes(types)
    actual_typename = type(current).__name__
    if actual_typename not in allowed_types:
        raise exceptions.JMESPathTypeError(function_name, current, self._convert_to_jmespath_type(actual_typename), types)
    if allowed_subtypes:
        self._subtype_check(current, allowed_subtypes, types, function_name)