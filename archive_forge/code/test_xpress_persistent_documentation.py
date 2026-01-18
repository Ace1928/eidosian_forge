import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.core.expr.taylor_series import taylor_series_expansion
from pyomo.solvers.plugins.solvers.xpress_direct import xpress_available
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
Test non-convex QP which xpress_direct should prove infeasible.