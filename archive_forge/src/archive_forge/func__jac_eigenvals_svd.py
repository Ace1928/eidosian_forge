from __future__ import absolute_import, division, print_function
import copy
import os
import warnings
from collections import defaultdict
import numpy as np
from .plotting import plot_result, plot_phase_plane
from .results import Result
from .util import _ensure_4args, _default
def _jac_eigenvals_svd(self, xval, yvals, intern_p):
    from scipy.linalg import svd
    J = self.j_cb(xval, yvals, intern_p)
    return svd(J, compute_uv=False)