from collections import namedtuple
from pyomo.core.base.objective import Objective
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.modeling import unique_component_name
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.opt import SolverResults, TerminationCondition
from pyomo.common.dependencies import (
class SecantNewtonNlpSolver(NewtonNlpSolver):
    """A wrapper around the SciPy scalar Newton solver for NLP objects
    that takes a specified number of secant iterations (default is 2) to
    try to converge a linear equation quickly then switches to Newton's
    method if this is not successful. This strategy is inspired by
    calculate_variable_from_constraint in pyomo.util.calc_var_value.

    """
    OPTIONS = ConfigBlock(description='Options for the SciPy Newton-secant hybrid')
    OPTIONS.declare_from(NewtonNlpSolver.OPTIONS, skip={'maxiter', 'secant'})
    OPTIONS.declare('secant_iter', ConfigValue(default=2, domain=int, description="Number of secant iterations to perform before switching to Newton's method."))
    OPTIONS.declare('newton_iter', ConfigValue(default=50, domain=int, description='Maximum iterations for the Newton solve'))

    def __init__(self, nlp, timer=None, options=None):
        super().__init__(nlp, timer=timer, options=options)
        self.converged_with_secant = None

    def solve(self, x0=None):
        if x0 is None:
            x0 = self._nlp.get_primals()
        try:
            results = sp.optimize.newton(lambda x: self.evaluate_function(np.array([x]))[0], x0[0], fprime=None, tol=self.options.tol, maxiter=self.options.secant_iter, full_output=self.options.full_output)
            self.converged_with_secant = True
        except RuntimeError:
            self.converged_with_secant = False
            x0 = self._nlp.get_primals()
            results = sp.optimize.newton(lambda x: self.evaluate_function(np.array([x]))[0], x0[0], fprime=lambda x: self.evaluate_jacobian(np.array([x]))[0, 0], tol=self.options.tol, maxiter=self.options.newton_iter, full_output=self.options.full_output)
        return results