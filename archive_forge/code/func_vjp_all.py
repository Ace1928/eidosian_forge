from __future__ import absolute_import
from builtins import range
import scipy.integrate
import autograd.numpy as np
from autograd.extend import primitive, defvjp_argnums
from autograd import make_vjp
from autograd.misc import flatten
from autograd.builtins import tuple
def vjp_all(g):
    vjp_y = g[-1, :]
    vjp_t0 = 0
    time_vjp_list = []
    vjp_args = np.zeros(np.size(flat_args))
    for i in range(T - 1, 0, -1):
        vjp_cur_t = np.dot(func(yt[i, :], t[i], *func_args), g[i, :])
        time_vjp_list.append(vjp_cur_t)
        vjp_t0 = vjp_t0 - vjp_cur_t
        aug_y0 = np.hstack((yt[i, :], vjp_y, vjp_t0, vjp_args))
        aug_ans = odeint(augmented_dynamics, aug_y0, np.array([t[i], t[i - 1]]), tuple((flat_args,)), **kwargs)
        _, vjp_y, vjp_t0, vjp_args = unpack(aug_ans[1])
        vjp_y = vjp_y + g[i - 1, :]
    time_vjp_list.append(vjp_t0)
    vjp_times = np.hstack(time_vjp_list)[::-1]
    return (None, vjp_y, vjp_times, unflatten(vjp_args))