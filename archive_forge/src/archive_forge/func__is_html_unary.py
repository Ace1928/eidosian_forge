import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def _is_html_unary(k):
    global _html_unary_map
    return _html_unary_map.get(k, False)