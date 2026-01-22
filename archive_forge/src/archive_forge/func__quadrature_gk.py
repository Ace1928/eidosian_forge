import sys
import copy
import heapq
import collections
import functools
import numpy as np
from scipy._lib._util import MapWrapper, _FunctionWrapper
def _quadrature_gk(a, b, f, norm_func, x, w, v):
    """
    Generic Gauss-Kronrod quadrature
    """
    fv = [0.0] * len(x)
    c = 0.5 * (a + b)
    h = 0.5 * (b - a)
    s_k = 0.0
    s_k_abs = 0.0
    for i in range(len(x)):
        ff = f(c + h * x[i])
        fv[i] = ff
        vv = v[i]
        s_k += vv * ff
        s_k_abs += vv * abs(ff)
    s_g = 0.0
    for i in range(len(w)):
        s_g += w[i] * fv[2 * i + 1]
    s_k_dabs = 0.0
    y0 = s_k / 2.0
    for i in range(len(x)):
        s_k_dabs += v[i] * abs(fv[i] - y0)
    err = float(norm_func((s_k - s_g) * h))
    dabs = float(norm_func(s_k_dabs * h))
    if dabs != 0 and err != 0:
        err = dabs * min(1.0, (200 * err / dabs) ** 1.5)
    eps = sys.float_info.epsilon
    round_err = float(norm_func(50 * eps * h * s_k_abs))
    if round_err > sys.float_info.min:
        err = max(err, round_err)
    return (h * s_k, err, round_err)