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
def AllChar(regex_sort, ctx=None):
    """Create a regular expression that accepts all single character strings
    """
    return ReRef(Z3_mk_re_allchar(regex_sort.ctx_ref(), regex_sort.ast), regex_sort.ctx)