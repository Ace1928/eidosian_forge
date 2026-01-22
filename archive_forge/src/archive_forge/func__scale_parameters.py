import warnings
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper
from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
from scipy.sparse import issparse
def _scale_parameters(self, trial):
    """Scale from a number between 0 and 1 to parameters."""
    scaled = self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2
    if np.any(self.integrality):
        i = np.broadcast_to(self.integrality, scaled.shape)
        scaled[i] = np.round(scaled[i])
    return scaled