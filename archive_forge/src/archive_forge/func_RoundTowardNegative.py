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
def RoundTowardNegative(ctx=None):
    ctx = _get_ctx(ctx)
    return FPRMRef(Z3_mk_fpa_round_toward_negative(ctx.ref()), ctx)