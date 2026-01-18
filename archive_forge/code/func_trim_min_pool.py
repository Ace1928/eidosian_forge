from collections import namedtuple
import time
import logging
import warnings
import sys
import numpy as np
from scipy import spatial
from scipy.optimize import OptimizeResult, minimize, Bounds
from scipy.optimize._optimize import MemoizeJac
from scipy.optimize._constraints import new_bounds_to_old
from scipy.optimize._minimize import standardize_constraints
from scipy._lib._util import _FunctionWrapper
from scipy.optimize._shgo_lib._complex import Complex
def trim_min_pool(self, trim_ind):
    self.X_min = np.delete(self.X_min, trim_ind, axis=0)
    self.minimizer_pool_F = np.delete(self.minimizer_pool_F, trim_ind)
    self.minimizer_pool = np.delete(self.minimizer_pool, trim_ind)
    return