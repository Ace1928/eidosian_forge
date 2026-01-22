from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.dependencies import attempt_import, numpy as np
from pyomo.core.base.objective import Objective
from pyomo.core.base.suffix import Suffix
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.subsystems import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import CyIpoptNLP
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.incidence_analysis.scc_solver import (
class ScipySolverWrapper(NlpSolverBase):
    """A wrapper for SciPy NLP solvers that implements the NlpSolverBase API

    This solver uses scipy.optimize.fsolve for "vector-valued" NLPs (with more
    than one primal variable and equality constraint) and the Secant-Newton
    hybrid for "scalar-valued" NLPs.

    """

    def __init__(self, nlp, timer=None, options=None):
        if options is None:
            options = {}
        for key in options:
            if key not in SecantNewtonNlpSolver.OPTIONS and key not in FsolveNlpSolver.OPTIONS:
                raise ValueError('Option %s is invalid for both SecantNewtonNlpSolver and FsolveNlpSolver' % key)
        newton_options = {key: value for key, value in options.items() if key in SecantNewtonNlpSolver.OPTIONS}
        fsolve_options = {key: value for key, value in options.items() if key in FsolveNlpSolver.OPTIONS}
        if nlp.n_primals() == 1:
            solver = SecantNewtonNlpSolver(nlp, timer=timer, options=newton_options)
        else:
            solver = FsolveNlpSolver(nlp, timer=timer, options=fsolve_options)
        self._nlp = nlp
        self._options = options
        self._solver = solver

    def solve(self, x0=None):
        res = self._solver.solve(x0=x0)
        return res