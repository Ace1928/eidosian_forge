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
def PiecewiseLinearOrder(a, index):
    return FuncDeclRef(Z3_mk_piecewise_linear_order(a.ctx_ref(), a.ast, index), a.ctx)