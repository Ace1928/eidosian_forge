import numpy as np
import scipy.sparse as sps
@classmethod
def _less_to_canonical(cls, cfun, ub, keep_feasible):
    empty_fun = np.empty(0)
    n = cfun.n
    if cfun.sparse_jacobian:
        empty_jac = sps.csr_matrix((0, n))
    else:
        empty_jac = np.empty((0, n))
    finite_ub = ub < np.inf
    n_eq = 0
    n_ineq = np.sum(finite_ub)
    if np.all(finite_ub):

        def fun(x):
            return (empty_fun, cfun.fun(x) - ub)

        def jac(x):
            return (empty_jac, cfun.jac(x))

        def hess(x, v_eq, v_ineq):
            return cfun.hess(x, v_ineq)
    else:
        finite_ub = np.nonzero(finite_ub)[0]
        keep_feasible = keep_feasible[finite_ub]
        ub = ub[finite_ub]

        def fun(x):
            return (empty_fun, cfun.fun(x)[finite_ub] - ub)

        def jac(x):
            return (empty_jac, cfun.jac(x)[finite_ub])

        def hess(x, v_eq, v_ineq):
            v = np.zeros(cfun.m)
            v[finite_ub] = v_ineq
            return cfun.hess(x, v)
    return cls(n_eq, n_ineq, fun, jac, hess, keep_feasible)