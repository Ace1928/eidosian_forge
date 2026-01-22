import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
@signature({'types': ['string']}, {'types': ['array-string']})
def _func_join(self, separator, array):
    return separator.join(array)