import warnings
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper
from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
from scipy.sparse import issparse
def _rand2(self, samples):
    """rand2bin, rand2exp"""
    r0, r1, r2, r3, r4 = samples
    bprime = self.population[r0] + self.scale * (self.population[r1] + self.population[r2] - self.population[r3] - self.population[r4])
    return bprime