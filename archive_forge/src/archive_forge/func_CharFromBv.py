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
def CharFromBv(bv):
    if not is_expr(bv):
        raise Z3Exception('Bit-vector expression needed')
    return _to_expr_ref(Z3_mk_char_from_bv(bv.ctx_ref(), bv.as_ast()), bv.ctx)