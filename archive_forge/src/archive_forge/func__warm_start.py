from pyomo.core.expr.numvalue import value
from pyomo.solvers.plugins.solvers.cplex_direct import CPLEXDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.opt.base import SolverFactory
def _warm_start(self):
    CPLEXDirect._warm_start(self)