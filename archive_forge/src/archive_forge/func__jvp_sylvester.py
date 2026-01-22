from __future__ import division
import scipy.linalg
import autograd.numpy as anp
from autograd.numpy.numpy_wrapper import wrap_namespace
from autograd.extend import defvjp, defvjp_argnums, defjvp, defjvp_argnums
def _jvp_sylvester(argnums, dms, ans, args, _):
    a, b, q = args
    if 0 in argnums:
        da = dms[0]
        db = dms[1] if 1 in argnums else 0
    else:
        da = 0
        db = dms[0] if 1 in argnums else 0
    dq = dms[-1] if 2 in argnums else 0
    rhs = dq - anp.dot(da, ans) - anp.dot(ans, db)
    return solve_sylvester(a, b, rhs)