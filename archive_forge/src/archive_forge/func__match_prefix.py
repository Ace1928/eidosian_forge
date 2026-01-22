import logging
import re
from .compat import string_types
from .util import parse_requirement
def _match_prefix(x, y):
    x = str(x)
    y = str(y)
    if x == y:
        return True
    if not x.startswith(y):
        return False
    n = len(y)
    return x[n] == '.'