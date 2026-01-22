import inspect
import numpy as np
from ._optimize import OptimizeWarning, OptimizeResult
from warnings import warn
from ._highs._highs_wrapper import _highs_wrapper
from ._highs._highs_constants import (
from scipy.sparse import csc_matrix, vstack, issparse
def _replace_inf(x):
    infs = np.isinf(x)
    with np.errstate(invalid='ignore'):
        x[infs] = np.sign(x[infs]) * CONST_INF
    return x