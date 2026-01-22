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
def TransitiveClosure(f):
    """Given a binary relation R, such that the two arguments have the same sort
    create the transitive closure relation R+.
    The transitive closure R+ is a new relation.
    """
    return FuncDeclRef(Z3_mk_transitive_closure(f.ctx_ref(), f.ast), f.ctx)