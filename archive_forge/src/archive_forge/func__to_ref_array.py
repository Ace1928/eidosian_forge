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
def _to_ref_array(ref, args):
    sz = len(args)
    _args = (ref * sz)()
    for i in range(sz):
        _args[i] = args[i].as_ast()
    return (_args, sz)