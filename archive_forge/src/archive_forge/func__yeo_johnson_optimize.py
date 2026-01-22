import warnings
from numbers import Integral, Real
import numpy as np
from scipy import optimize, sparse, stats
from scipy.special import boxcox
from ..base import (
from ..utils import _array_api, check_array
from ..utils._array_api import get_namespace
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.extmath import _incremental_mean_and_var, row_norms
from ..utils.sparsefuncs import (
from ..utils.sparsefuncs_fast import (
from ..utils.validation import (
from ._encoders import OneHotEncoder
def _yeo_johnson_optimize(self, x):
    """Find and return optimal lambda parameter of the Yeo-Johnson
        transform by MLE, for observed data x.

        Like for Box-Cox, MLE is done via the brent optimizer.
        """
    x_tiny = np.finfo(np.float64).tiny

    def _neg_log_likelihood(lmbda):
        """Return the negative log likelihood of the observed data x as a
            function of lambda."""
        x_trans = self._yeo_johnson_transform(x, lmbda)
        n_samples = x.shape[0]
        x_trans_var = x_trans.var()
        if x_trans_var < x_tiny:
            return np.inf
        log_var = np.log(x_trans_var)
        loglike = -n_samples / 2 * log_var
        loglike += (lmbda - 1) * (np.sign(x) * np.log1p(np.abs(x))).sum()
        return -loglike
    x = x[~np.isnan(x)]
    return optimize.brent(_neg_log_likelihood, brack=(-2, 2))