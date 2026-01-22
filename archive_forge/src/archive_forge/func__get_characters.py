import copy
import re
from collections import namedtuple
def _get_characters(self, name, default_value=''):
    option_value = getattr(self.raw_options, name, default_value)
    result = ''
    if isinstance(option_value, str):
        result = option_value.replace('\\r', '\r').replace('\\n', '\n').replace('\\t', '\t')
    return result