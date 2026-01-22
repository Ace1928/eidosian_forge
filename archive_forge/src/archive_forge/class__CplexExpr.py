import logging
import re
import sys
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.base import Suffix, Var, Constraint, SOSConstraint, Objective
from pyomo.core.expr.numvalue import is_fixed
from pyomo.core.expr.numvalue import value
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn import generate_standard_repn
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
from pyomo.opt.base import SolverFactory
import time
class _CplexExpr(object):

    def __init__(self, variables, coefficients, offset=None, q_variables1=None, q_variables2=None, q_coefficients=None):
        self.variables = variables
        self.coefficients = coefficients
        self.offset = offset or 0.0
        self.q_variables1 = q_variables1 or []
        self.q_variables2 = q_variables2 or []
        self.q_coefficients = [float(coef) for coef in q_coefficients or []]