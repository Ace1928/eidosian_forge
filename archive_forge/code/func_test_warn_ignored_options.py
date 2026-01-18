import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from scipy.optimize import (NonlinearConstraint, LinearConstraint,
from .test_minimize_constrained import (Maratos, HyperbolicIneq, Rosenbrock,
def test_warn_ignored_options(self):

    def fun(x):
        return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2 + (x[2] - 0.75) ** 2
    x0 = (2, 0, 1)
    if self.method == 'slsqp':
        bnds = ((0, None), (0, None), (0, None))
    else:
        bnds = None
    cons = NonlinearConstraint(lambda x: x[0], 2, np.inf)
    res = minimize(fun, x0, method=self.method, bounds=bnds, constraints=cons)
    assert_allclose(res.fun, 1)
    cons = LinearConstraint([1, 0, 0], 2, np.inf)
    res = minimize(fun, x0, method=self.method, bounds=bnds, constraints=cons)
    assert_allclose(res.fun, 1)
    cons = []
    cons.append(NonlinearConstraint(lambda x: x[0] ** 2, 2, np.inf, keep_feasible=True))
    cons.append(NonlinearConstraint(lambda x: x[0] ** 2, 2, np.inf, hess=BFGS()))
    cons.append(NonlinearConstraint(lambda x: x[0] ** 2, 2, np.inf, finite_diff_jac_sparsity=42))
    cons.append(NonlinearConstraint(lambda x: x[0] ** 2, 2, np.inf, finite_diff_rel_step=42))
    cons.append(LinearConstraint([1, 0, 0], 2, np.inf, keep_feasible=True))
    for con in cons:
        assert_warns(OptimizeWarning, minimize, fun, x0, method=self.method, bounds=bnds, constraints=cons)