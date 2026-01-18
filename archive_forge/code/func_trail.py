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
def trail(self):
    """Return trail of the solver state after a check() call.
        """
    return AstVector(Z3_solver_get_trail(self.ctx.ref(), self.solver), self.ctx)