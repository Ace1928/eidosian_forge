import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
def _get_allowed_pytypes(self, types):
    allowed_types = []
    allowed_subtypes = []
    for t in types:
        type_ = t.split('-', 1)
        if len(type_) == 2:
            type_, subtype = type_
            allowed_subtypes.append(REVERSE_TYPES_MAP[subtype])
        else:
            type_ = type_[0]
        allowed_types.extend(REVERSE_TYPES_MAP[type_])
    return (allowed_types, allowed_subtypes)