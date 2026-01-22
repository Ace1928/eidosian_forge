import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from scipy.optimize import (NonlinearConstraint, LinearConstraint,
from .test_minimize_constrained import (Maratos, HyperbolicIneq, Rosenbrock,
class TestOldToNew:
    x0 = (2, 0)
    bnds = ((0, None), (0, None))
    method = 'trust-constr'

    def test_constraint_dictionary_1(self):

        def fun(x):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
        cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2}, {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6}, {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'delta_grad == 0.0')
            res = minimize(fun, self.x0, method=self.method, bounds=self.bnds, constraints=cons)
        assert_allclose(res.x, [1.4, 1.7], rtol=0.0001)
        assert_allclose(res.fun, 0.8, rtol=0.0001)

    def test_constraint_dictionary_2(self):

        def fun(x):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
        cons = {'type': 'eq', 'fun': lambda x, p1, p2: p1 * x[0] - p2 * x[1], 'args': (1, 1.1), 'jac': lambda x, p1, p2: np.array([[p1, -p2]])}
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'delta_grad == 0.0')
            res = minimize(fun, self.x0, method=self.method, bounds=self.bnds, constraints=cons)
        assert_allclose(res.x, [1.7918552, 1.62895927])
        assert_allclose(res.fun, 1.3857466063348418)

    def test_constraint_dictionary_3(self):

        def fun(x):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
        cons = [{'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2}, NonlinearConstraint(lambda x: x[0] - x[1], 0, 0)]
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'delta_grad == 0.0')
            res = minimize(fun, self.x0, method=self.method, bounds=self.bnds, constraints=cons)
        assert_allclose(res.x, [1.75, 1.75], rtol=0.0001)
        assert_allclose(res.fun, 1.125, rtol=0.0001)