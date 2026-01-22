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
def _ast_kind(ctx, a):
    if is_ast(a):
        a = a.as_ast()
    return Z3_get_ast_kind(ctx.ref(), a)