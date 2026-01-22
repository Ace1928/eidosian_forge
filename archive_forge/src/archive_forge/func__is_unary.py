import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def _is_unary(k):
    global _unary_map
    return _unary_map.get(k, False)