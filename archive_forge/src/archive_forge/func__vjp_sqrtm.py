from __future__ import division
import scipy.linalg
import autograd.numpy as anp
from autograd.numpy.numpy_wrapper import wrap_namespace
from autograd.extend import defvjp, defvjp_argnums, defjvp, defjvp_argnums
def _vjp_sqrtm(ans, A, disp=True, blocksize=64):
    assert disp, 'sqrtm vjp not implemented for disp=False'
    ans_transp = anp.transpose(ans)

    def vjp(g):
        return anp.real(solve_sylvester(ans_transp, ans_transp, g))
    return vjp