import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def _is_html_assoc(k):
    return k == Z3_OP_AND or k == Z3_OP_OR or k == Z3_OP_IFF or _is_assoc(k)