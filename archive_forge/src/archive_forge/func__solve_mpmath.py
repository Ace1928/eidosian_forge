from __future__ import absolute_import, division, print_function
import inspect
import math
import os
import warnings
import sys
import numpy as np
from .plotting import plot_series
def _solve_mpmath(self, intern_x0, dps=30, tol=None, maxsteps=None, **kwargs):
    import mpmath
    from mpmath.calculus.optimization import MDNewton
    mp = mpmath.mp
    mp.dps = dps

    def _mpf(val):
        try:
            return mp.mpf(val)
        except TypeError:
            return mp.mpf(float(val))
    intern_p = tuple((_mpf(_p) for _p in self.internal_params))
    maxsteps = maxsteps or MDNewton.maxsteps
    tol = tol or mp.eps * 1024

    def f_cb(*x):
        f_cb.nfev += 1
        return self.f_cb(x, intern_p)
    f_cb.nfev = 0
    if self.j_cb is not None:

        def j_cb(*x):
            j_cb.njev += 1
            return self.j_cb(x, intern_p)
        j_cb.njev = 0
        kwargs['J'] = j_cb
    intern_x0 = tuple((_mpf(_x) for _x in intern_x0))
    iters = MDNewton(mp, f_cb, intern_x0, norm=mp.norm, verbose=False, **kwargs)
    i = 0
    success = False
    for x, err in iters:
        i += 1
        lim = tol * max(mp.norm(x), 1)
        if err < lim:
            success = True
            break
        if i >= maxsteps:
            break
    result = {'x': x, 'success': success, 'nfev': f_cb.nfev, 'nit': i}
    if self.j_cb is not None:
        result['njev'] = j_cb.njev
    return result