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
def Unit(a):
    """Create a singleton sequence"""
    return SeqRef(Z3_mk_seq_unit(a.ctx_ref(), a.as_ast()), a.ctx)