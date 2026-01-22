from __future__ import absolute_import, division, print_function
import inspect
import math
import os
import warnings
import sys
import numpy as np
from .plotting import plot_series
def _solve_nleq2(self, intern_x0, tol=1e-08, method=None, **kwargs):
    from pynleq2 import solve

    def f_cb(x, ierr):
        f_cb.nfev += 1
        return (self.f_cb(x, self.internal_params), ierr)
    f_cb.nfev = 0

    def j_cb(x, ierr):
        j_cb.njev += 1
        return (self.j_cb(x, self.internal_params), ierr)
    j_cb.njev = 0
    x, ierr = solve(f_cb, j_cb, intern_x0, **kwargs)
    return {'x': x, 'fun': np.asarray(f_cb(x, 0)), 'success': ierr == 0, 'nfev': f_cb.nfev, 'njev': j_cb.njev, 'ierr': ierr}