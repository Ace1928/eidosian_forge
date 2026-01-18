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
def trail_levels(self):
    """Return trail and decision levels of the solver state after a check() call.
        """
    trail = self.trail()
    levels = (ctypes.c_uint * len(trail))()
    Z3_solver_get_levels(self.ctx.ref(), self.solver, trail.vector, len(trail), levels)
    return (trail, levels)