from statsmodels.compat.python import lrange
import math
import scipy.stats
import numpy as np
from scipy.optimize import fminbound
def _interpolate_p(p, r, v):
    """
    interpolates p based on the values in the A table for the
    scalar value of r and the scalar value of v
    """
    p0, p1, p2 = _select_ps(p)
    try:
        y0 = _func(A[p0, v], p0, r, v) + 1.0
    except:
        print(p, r, v)
        raise
    y1 = _func(A[p1, v], p1, r, v) + 1.0
    y2 = _func(A[p2, v], p2, r, v) + 1.0
    y_log0 = math.log(y0 + float(r) / float(v))
    y_log1 = math.log(y1 + float(r) / float(v))
    y_log2 = math.log(y2 + float(r) / float(v))
    if p > 0.85:
        p_t = _ptransform(p)
        p0_t = _ptransform(p0)
        p1_t = _ptransform(p1)
        p2_t = _ptransform(p2)
        d2 = 2 * ((y_log2 - y_log1) / (p2_t - p1_t) - (y_log1 - y_log0) / (p1_t - p0_t)) / (p2_t - p0_t)
        if p2 + p0 >= p1 + p1:
            d1 = (y_log2 - y_log1) / (p2_t - p1_t) - 0.5 * d2 * (p2_t - p1_t)
        else:
            d1 = (y_log1 - y_log0) / (p1_t - p0_t) + 0.5 * d2 * (p1_t - p0_t)
        d0 = y_log1
        y_log = d2 / 2.0 * (p_t - p1_t) ** 2.0 + d1 * (p_t - p1_t) + d0
        y = math.exp(y_log) - float(r) / float(v)
    elif p > 0.5:
        d2 = 2 * ((y_log2 - y_log1) / (p2 - p1) - (y_log1 - y_log0) / (p1 - p0)) / (p2 - p0)
        if p2 + p0 >= p1 + p1:
            d1 = (y_log2 - y_log1) / (p2 - p1) - 0.5 * d2 * (p2 - p1)
        else:
            d1 = (y_log1 - y_log0) / (p1 - p0) + 0.5 * d2 * (p1 - p0)
        d0 = y_log1
        y_log = d2 / 2.0 * (p - p1) ** 2.0 + d1 * (p - p1) + d0
        y = math.exp(y_log) - float(r) / float(v)
    else:
        v = min(v, 1e+38)
        q0 = math.sqrt(2) * -y0 * scipy.stats.t.isf((1.0 + p0) / 2.0, v)
        q1 = math.sqrt(2) * -y1 * scipy.stats.t.isf((1.0 + p1) / 2.0, v)
        d1 = (q1 - q0) / (p1 - p0)
        d0 = q0
        q = d1 * (p - p0) + d0
        y = -q / (math.sqrt(2) * scipy.stats.t.isf((1.0 + p) / 2.0, v))
    return y