import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from scipy.optimize._constraints import (NonlinearConstraint, Bounds,
from scipy.optimize._trustregion_constr.canonical_constraint \
def test_nonlinear_constraint():
    n = 3
    m = 5
    rng = np.random.RandomState(0)
    x0 = rng.rand(n)
    fun, jac, hess = create_quadratic_function(n, m, rng)
    f = fun(x0)
    J = jac(x0)
    lb = [-10, 3, -np.inf, -np.inf, -5]
    ub = [10, 3, np.inf, 3, np.inf]
    user_constraint = NonlinearConstraint(fun, lb, ub, jac, hess, [True, False, False, True, False])
    for sparse_jacobian in [False, True]:
        prepared_constraint = PreparedConstraint(user_constraint, x0, sparse_jacobian)
        c = CanonicalConstraint.from_PreparedConstraint(prepared_constraint)
        assert_array_equal(c.n_eq, 1)
        assert_array_equal(c.n_ineq, 4)
        c_eq, c_ineq = c.fun(x0)
        assert_array_equal(c_eq, [f[1] - lb[1]])
        assert_array_equal(c_ineq, [f[3] - ub[3], lb[4] - f[4], f[0] - ub[0], lb[0] - f[0]])
        J_eq, J_ineq = c.jac(x0)
        if sparse_jacobian:
            J_eq = J_eq.toarray()
            J_ineq = J_ineq.toarray()
        assert_array_equal(J_eq, J[1, None])
        assert_array_equal(J_ineq, np.vstack((J[3], -J[4], J[0], -J[0])))
        v_eq = rng.rand(c.n_eq)
        v_ineq = rng.rand(c.n_ineq)
        v = np.zeros(m)
        v[1] = v_eq[0]
        v[3] = v_ineq[0]
        v[4] = -v_ineq[1]
        v[0] = v_ineq[2] - v_ineq[3]
        assert_array_equal(c.hess(x0, v_eq, v_ineq), hess(x0, v))
        assert_array_equal(c.keep_feasible, [True, False, True, True])