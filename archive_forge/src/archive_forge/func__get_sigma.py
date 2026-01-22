from __future__ import annotations
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lrange, lzip
from typing import Literal
from collections.abc import Sequence
import warnings
import numpy as np
from scipy import optimize, stats
from scipy.linalg import cholesky, toeplitz
from scipy.linalg.lapack import dtrtri
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.emplike.elregress import _ELRegOpts
from statsmodels.regression._prediction import PredictionResults
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import pinv_extended
from statsmodels.tools.typing import Float64Array
from statsmodels.tools.validation import bool_like, float_like, string_like
from . import _prediction as pred
def _get_sigma(sigma, nobs):
    """
    Returns sigma (matrix, nobs by nobs) for GLS and the inverse of its
    Cholesky decomposition.  Handles dimensions and checks integrity.
    If sigma is None, returns None, None. Otherwise returns sigma,
    cholsigmainv.
    """
    if sigma is None:
        return (None, None)
    sigma = np.asarray(sigma).squeeze()
    if sigma.ndim == 0:
        sigma = np.repeat(sigma, nobs)
    if sigma.ndim == 1:
        if sigma.shape != (nobs,):
            raise ValueError('Sigma must be a scalar, 1d of length %s or a 2d array of shape %s x %s' % (nobs, nobs, nobs))
        cholsigmainv = 1 / np.sqrt(sigma)
    else:
        if sigma.shape != (nobs, nobs):
            raise ValueError('Sigma must be a scalar, 1d of length %s or a 2d array of shape %s x %s' % (nobs, nobs, nobs))
        cholsigmainv, info = dtrtri(cholesky(sigma, lower=True), lower=True, overwrite_c=True)
        if info > 0:
            raise np.linalg.LinAlgError('Cholesky decomposition of sigma yields a singular matrix')
        elif info < 0:
            raise ValueError('Invalid input to dtrtri (info = %d)' % info)
    return (sigma, cholsigmainv)