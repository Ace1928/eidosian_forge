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
def Complement(re):
    """Create the complement regular expression."""
    return ReRef(Z3_mk_re_complement(re.ctx_ref(), re.as_ast()), re.ctx)