from collections import namedtuple
from pyomo.core.base.objective import Objective
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.modeling import unique_component_name
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.opt import SolverResults, TerminationCondition
from pyomo.common.dependencies import (
class FsolveNlpSolver(DenseSquareNlpSolver):
    OPTIONS = DenseSquareNlpSolver.OPTIONS(description='Options for SciPy fsolve wrapper')
    OPTIONS.declare('xtol', ConfigValue(default=1e-08, domain=float, description='Tolerance for convergence of variable vector'))
    OPTIONS.declare('maxfev', ConfigValue(default=100, domain=int, description='Maximum number of function evaluations per solve'))
    OPTIONS.declare('tol', ConfigValue(default=None, domain=float, description='Tolerance for convergence of function residual'))
    OPTIONS.declare('full_output', ConfigValue(default=True, domain=bool))

    def solve(self, x0=None):
        if x0 is None:
            x0 = self._nlp.get_primals()
        res = sp.optimize.fsolve(self.evaluate_function, x0, fprime=self.evaluate_jacobian, full_output=self.options.full_output, xtol=self.options.xtol, maxfev=self.options.maxfev)
        if self.options.full_output:
            x, info, ier, msg = res
        else:
            x, ier, msg = res
        if self.options.tol is not None:
            if self.options.full_output:
                fcn_val = info['fvec']
            else:
                fcn_val = self.evaluate_function(x)
            if not np.all(np.abs(fcn_val) <= self.options.tol):
                raise RuntimeError("fsolve converged to a solution that does not satisfy the function tolerance 'tol' of %s. You may need to relax the 'tol' option or tighten the 'xtol' option (currently 'xtol' is %s)." % (self.options.tol, self.options.xtol))
        return res