import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from scipy.optimize import (NonlinearConstraint, LinearConstraint,
from .test_minimize_constrained import (Maratos, HyperbolicIneq, Rosenbrock,
class TestNewToOld:

    def test_multiple_constraint_objects(self):

        def fun(x):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2 + (x[2] - 0.75) ** 2
        x0 = [2, 0, 1]
        coni = []
        methods = ['slsqp', 'cobyla', 'trust-constr']
        coni.append([{'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2}, NonlinearConstraint(lambda x: x[0] - x[1], -1, 1)])
        coni.append([LinearConstraint([1, -2, 0], -2, np.inf), NonlinearConstraint(lambda x: x[0] - x[1], -1, 1)])
        coni.append([NonlinearConstraint(lambda x: x[0] - 2 * x[1] + 2, 0, np.inf), NonlinearConstraint(lambda x: x[0] - x[1], -1, 1)])
        for con in coni:
            funs = {}
            for method in methods:
                with suppress_warnings() as sup:
                    sup.filter(UserWarning)
                    result = minimize(fun, x0, method=method, constraints=con)
                    funs[method] = result.fun
            assert_allclose(funs['slsqp'], funs['trust-constr'], rtol=0.0001)
            assert_allclose(funs['cobyla'], funs['trust-constr'], rtol=0.0001)

    def test_individual_constraint_objects(self):

        def fun(x):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2 + (x[2] - 0.75) ** 2
        x0 = [2, 0, 1]
        cone = []
        coni = []
        methods = ['slsqp', 'cobyla', 'trust-constr']
        cone.append(NonlinearConstraint(lambda x: x[0] - x[1], 1, 1))
        cone.append(NonlinearConstraint(lambda x: x[0] - x[1], [1.21], [1.21]))
        cone.append(NonlinearConstraint(lambda x: x[0] - x[1], 1.21, np.array([1.21])))
        cone.append(NonlinearConstraint(lambda x: [x[0] - x[1], x[1] - x[2]], 1.21, 1.21))
        cone.append(NonlinearConstraint(lambda x: [x[0] - x[1], x[1] - x[2]], [1.21, 1.4], [1.21, 1.4]))
        cone.append(NonlinearConstraint(lambda x: [x[0] - x[1], x[1] - x[2]], [1.21, 1.21], 1.21))
        cone.append(NonlinearConstraint(lambda x: [x[0] - x[1], x[1] - x[2]], [1.21, -np.inf], [1.21, np.inf]))
        coni.append(NonlinearConstraint(lambda x: x[0] - x[1], 1.21, np.inf))
        coni.append(NonlinearConstraint(lambda x: x[0] - x[1], [1.21], np.inf))
        coni.append(NonlinearConstraint(lambda x: x[0] - x[1], 1.21, np.array([np.inf])))
        coni.append(NonlinearConstraint(lambda x: x[0] - x[1], -np.inf, -3))
        coni.append(NonlinearConstraint(lambda x: x[0] - x[1], np.array(-np.inf), -3))
        coni.append(NonlinearConstraint(lambda x: [x[0] - x[1], x[1] - x[2]], 1.21, np.inf))
        cone.append(NonlinearConstraint(lambda x: [x[0] - x[1], x[1] - x[2]], [1.21, -np.inf], [1.21, 1.4]))
        coni.append(NonlinearConstraint(lambda x: [x[0] - x[1], x[1] - x[2]], [1.1, 0.8], [1.2, 1.4]))
        coni.append(NonlinearConstraint(lambda x: [x[0] - x[1], x[1] - x[2]], [-1.2, -1.4], [-1.1, -0.8]))
        cone.append(LinearConstraint([1, -1, 0], 1.21, 1.21))
        cone.append(LinearConstraint([[1, -1, 0], [0, 1, -1]], 1.21, 1.21))
        cone.append(LinearConstraint([[1, -1, 0], [0, 1, -1]], [1.21, -np.inf], [1.21, 1.4]))
        for con in coni:
            funs = {}
            for method in methods:
                with suppress_warnings() as sup:
                    sup.filter(UserWarning)
                    result = minimize(fun, x0, method=method, constraints=con)
                    funs[method] = result.fun
            assert_allclose(funs['slsqp'], funs['trust-constr'], rtol=0.001)
            assert_allclose(funs['cobyla'], funs['trust-constr'], rtol=0.001)
        for con in cone:
            funs = {}
            for method in methods[::2]:
                with suppress_warnings() as sup:
                    sup.filter(UserWarning)
                    result = minimize(fun, x0, method=method, constraints=con)
                    funs[method] = result.fun
            assert_allclose(funs['slsqp'], funs['trust-constr'], rtol=0.001)