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

            This code in this if statement is only needed for backwards compatibility. It is more efficient to set
            _save_results to False and use load_vars, load_duals, etc.
            