import copy
import re
from collections import namedtuple
def _get_boolean(self, name, default_value=False):
    option_value = getattr(self.raw_options, name, default_value)
    result = False
    try:
        result = bool(option_value)
    except ValueError:
        pass
    return result