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
def _get_args_ast_list(args):
    try:
        if isinstance(args, (set, AstVector, tuple)):
            return [arg for arg in args]
        else:
            return args
    except Exception:
        return args