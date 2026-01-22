import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import numpy as np, scipy_available, numpy_available
from pyomo.common.log import LoggingIntercept
from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler
def _verify_solution(self, soln, repn, eq):
    for v, val in soln:
        v.value = None
    for v in repn.x:
        v.value = None
    x = np.array(repn.x, dtype=object)
    ax = repn.A.todense() @ x

    def c_rule(m, i):
        if eq:
            return ax[i] == repn.b[i]
        else:
            return ax[i] <= repn.b[i]
    test_model = pyo.ConcreteModel()
    test_model.o = pyo.Objective(expr=repn.c[[1], :].todense()[0] @ x)
    test_model.c = pyo.Constraint(range(len(repn.b)), rule=c_rule)
    linear_solver.solve(test_model, tee=True)
    for v, expr in repn.eliminated_vars:
        v.value = pyo.value(expr)
    self.assertEqual(*zip(*((v.value, val) for v, val in soln)))