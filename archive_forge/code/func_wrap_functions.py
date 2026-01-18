from warnings import warn
import numpy as np
from numpy.linalg import pinv
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import OptimizeResult
def wrap_functions(fun, bc, fun_jac, bc_jac, k, a, S, D, dtype):
    """Wrap functions for unified usage in the solver."""
    if fun_jac is None:
        fun_jac_wrapped = None
    if bc_jac is None:
        bc_jac_wrapped = None
    if k == 0:

        def fun_p(x, y, _):
            return np.asarray(fun(x, y), dtype)

        def bc_wrapped(ya, yb, _):
            return np.asarray(bc(ya, yb), dtype)
        if fun_jac is not None:

            def fun_jac_p(x, y, _):
                return (np.asarray(fun_jac(x, y), dtype), None)
        if bc_jac is not None:

            def bc_jac_wrapped(ya, yb, _):
                dbc_dya, dbc_dyb = bc_jac(ya, yb)
                return (np.asarray(dbc_dya, dtype), np.asarray(dbc_dyb, dtype), None)
    else:

        def fun_p(x, y, p):
            return np.asarray(fun(x, y, p), dtype)

        def bc_wrapped(x, y, p):
            return np.asarray(bc(x, y, p), dtype)
        if fun_jac is not None:

            def fun_jac_p(x, y, p):
                df_dy, df_dp = fun_jac(x, y, p)
                return (np.asarray(df_dy, dtype), np.asarray(df_dp, dtype))
        if bc_jac is not None:

            def bc_jac_wrapped(ya, yb, p):
                dbc_dya, dbc_dyb, dbc_dp = bc_jac(ya, yb, p)
                return (np.asarray(dbc_dya, dtype), np.asarray(dbc_dyb, dtype), np.asarray(dbc_dp, dtype))
    if S is None:
        fun_wrapped = fun_p
    else:

        def fun_wrapped(x, y, p):
            f = fun_p(x, y, p)
            if x[0] == a:
                f[:, 0] = np.dot(D, f[:, 0])
                f[:, 1:] += np.dot(S, y[:, 1:]) / (x[1:] - a)
            else:
                f += np.dot(S, y) / (x - a)
            return f
    if fun_jac is not None:
        if S is None:
            fun_jac_wrapped = fun_jac_p
        else:
            Sr = S[:, :, np.newaxis]

            def fun_jac_wrapped(x, y, p):
                df_dy, df_dp = fun_jac_p(x, y, p)
                if x[0] == a:
                    df_dy[:, :, 0] = np.dot(D, df_dy[:, :, 0])
                    df_dy[:, :, 1:] += Sr / (x[1:] - a)
                else:
                    df_dy += Sr / (x - a)
                return (df_dy, df_dp)
    return (fun_wrapped, bc_wrapped, fun_jac_wrapped, bc_jac_wrapped)