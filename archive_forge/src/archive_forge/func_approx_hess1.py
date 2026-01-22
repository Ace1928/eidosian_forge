import numpy as np
from statsmodels.compat.pandas import Appender, Substitution
@Substitution(scale='3', extra_params='return_grad : bool\n        Whether or not to also return the gradient\n', extra_returns='grad : nparray\n        Gradient if return_grad == True\n', equation_number='7', equation='1/(d_j*d_k) * ((f(x + d[j]*e[j] + d[k]*e[k]) - f(x + d[j]*e[j])))\n')
@Appender(_hessian_docs)
def approx_hess1(x, f, epsilon=None, args=(), kwargs={}, return_grad=False):
    n = len(x)
    h = _get_epsilon(x, 3, epsilon, n)
    ee = np.diag(h)
    f0 = f(*(x,) + args, **kwargs)
    g = np.zeros(n)
    for i in range(n):
        g[i] = f(*(x + ee[i, :],) + args, **kwargs)
    hess = np.outer(h, h)
    for i in range(n):
        for j in range(i, n):
            hess[i, j] = (f(*(x + ee[i, :] + ee[j, :],) + args, **kwargs) - g[i] - g[j] + f0) / hess[i, j]
            hess[j, i] = hess[i, j]
    if return_grad:
        grad = (g - f0) / h
        return (hess, grad)
    else:
        return hess