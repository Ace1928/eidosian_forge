import pyomo.common.unittest as unittest
from pyomo.common.errors import DeveloperError, NondifferentiableError
from pyomo.environ import (
from pyomo.core.expr.calculus.diff_with_sympy import differentiate
from pyomo.core.expr.sympy_tools import (
class bogus(object):

    def __init__(self):
        self.args = (obj_map.getSympySymbol(m.x),)
        self.func = type(self)