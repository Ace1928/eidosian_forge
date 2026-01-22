from __future__ import division
import scipy.linalg
import autograd.numpy as anp
from autograd.numpy.numpy_wrapper import wrap_namespace
from autograd.extend import defvjp, defvjp_argnums, defjvp, defjvp_argnums
def _vjp_sylvester(argnums, ans, args, _):
    a, b, q = args

    def vjp(g):
        vjps = []
        q_vjp = solve_sylvester(anp.transpose(a), anp.transpose(b), g)
        if 0 in argnums:
            vjps.append(-anp.dot(q_vjp, anp.transpose(ans)))
        if 1 in argnums:
            vjps.append(-anp.dot(anp.transpose(ans), q_vjp))
        if 2 in argnums:
            vjps.append(q_vjp)
        return tuple(vjps)
    return vjp