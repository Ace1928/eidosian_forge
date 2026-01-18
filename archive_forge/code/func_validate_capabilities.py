from os.path import join, dirname, abspath
import json
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.kernel.block import IBlock
from pyomo.core import Suffix, Var, Constraint, Objective
from pyomo.opt import ProblemFormat, SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
def validate_capabilities(self, opt):
    """Validate the capabilities of the optimizer"""
    if self.linear is True and (not opt.has_capability('linear') is True):
        return False
    if self.integer is True and (not opt.has_capability('integer') is True):
        return False
    if self.quadratic_objective is True and (not opt.has_capability('quadratic_objective') is True):
        return False
    if self.quadratic_constraint is True and (not opt.has_capability('quadratic_constraint') is True):
        return False
    if self.sos1 is True and (not opt.has_capability('sos1') is True):
        return False
    if self.sos2 is True and (not opt.has_capability('sos2') is True):
        return False
    return True