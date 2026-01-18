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
def using_params(self, *args, **keys):
    """Return a simplifier that uses the given configuration options"""
    p = args2params(args, keys, self.ctx)
    return Simplifier(Z3_simplifier_using_params(self.ctx.ref(), self.simplifier, p.params), self.ctx)