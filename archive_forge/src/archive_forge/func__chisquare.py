import warnings
from numbers import Integral, Real
import numpy as np
from scipy import special, stats
from scipy.sparse import issparse
from ..base import BaseEstimator, _fit_context
from ..preprocessing import LabelBinarizer
from ..utils import as_float_array, check_array, check_X_y, safe_mask, safe_sqr
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.validation import check_is_fitted
from ._base import SelectorMixin
def _chisquare(f_obs, f_exp):
    """Fast replacement for scipy.stats.chisquare.

    Version from https://github.com/scipy/scipy/pull/2525 with additional
    optimizations.
    """
    f_obs = np.asarray(f_obs, dtype=np.float64)
    k = len(f_obs)
    chisq = f_obs
    chisq -= f_exp
    chisq **= 2
    with np.errstate(invalid='ignore'):
        chisq /= f_exp
    chisq = chisq.sum(axis=0)
    return (chisq, special.chdtrc(k - 1, chisq))