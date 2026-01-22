import numpy as np
import scipy.sparse as sps
@classmethod
def _equal_to_canonical(cls, cfun, value):
    empty_fun = np.empty(0)
    n = cfun.n
    n_eq = value.shape[0]
    n_ineq = 0
    keep_feasible = np.empty(0, dtype=bool)
    if cfun.sparse_jacobian:
        empty_jac = sps.csr_matrix((0, n))
    else:
        empty_jac = np.empty((0, n))

    def fun(x):
        return (cfun.fun(x) - value, empty_fun)

    def jac(x):
        return (cfun.jac(x), empty_jac)

    def hess(x, v_eq, v_ineq):
        return cfun.hess(x, v_eq)
    empty_fun = np.empty(0)
    n = cfun.n
    if cfun.sparse_jacobian:
        empty_jac = sps.csr_matrix((0, n))
    else:
        empty_jac = np.empty((0, n))
    return cls(n_eq, n_ineq, fun, jac, hess, keep_feasible)