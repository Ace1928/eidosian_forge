import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def _is_infix(k):
    global _infix_map
    return _infix_map.get(k, False)