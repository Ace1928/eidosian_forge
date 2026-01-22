from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
def _R_compat_quantile(x, probs):
    probs = np.asarray(probs)
    quantiles = np.asarray([np.percentile(x, 100 * prob) for prob in probs.ravel(order='C')])
    return quantiles.reshape(probs.shape, order='C')