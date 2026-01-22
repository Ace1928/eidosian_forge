import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
@signature({'types': ['string']}, {'types': ['string']})
def _func_ends_with(self, search, suffix):
    return search.endswith(suffix)