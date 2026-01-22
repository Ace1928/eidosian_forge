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
def CharToInt(ch, ctx=None):
    ch = _coerce_char(ch, ctx)
    return ch.to_int()