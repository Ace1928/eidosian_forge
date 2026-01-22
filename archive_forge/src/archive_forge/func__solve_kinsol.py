from __future__ import absolute_import, division, print_function
import inspect
import math
import os
import warnings
import sys
import numpy as np
from .plotting import plot_series
def _solve_kinsol(self, intern_x0, **kwargs):
    import pykinsol

    def _f(x, fout):
        res = self.f_cb(x, self.internal_params)
        fout[:] = res

    def _j(x, Jout, fx):
        res = self.j_cb(x, self.internal_params)
        Jout[:, :] = res[:, :]
    return pykinsol.solve(_f, _j, intern_x0, **kwargs)