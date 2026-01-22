from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def _z3_check_cint_overflow(n, name):
    _z3_assert(ctypes.c_int(n).value == n, name + ' is too large')