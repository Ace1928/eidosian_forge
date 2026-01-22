from __future__ import absolute_import, print_function, division
import operator
from functools import partial
from petl.compat import text_type, binary_type, numeric_types
def _typestr(x):
    if isinstance(x, binary_type):
        return 'str'
    if isinstance(x, text_type):
        return 'unicode'
    return type(x).__name__