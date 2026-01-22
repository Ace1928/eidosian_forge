from collections import defaultdict
import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy
import scipy.stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.validation import string_like
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.coint_tables import c_sja, c_sjt
from statsmodels.tsa.tsatools import duplication_matrix, lagmat, vec
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
import statsmodels.tsa.vector_ar.irf as irf
import statsmodels.tsa.vector_ar.plotting as plot
from statsmodels.tsa.vector_ar.util import get_index, seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import (
def _sij(delta_x, delta_y_1_T, y_lag1):
    """Returns matrices and eigenvalues and -vectors used for parameter
    estimation and the calculation of a models loglikelihood.

    Parameters
    ----------
    delta_x : ndarray (k_ar_diff*neqs x nobs)
        (dimensions assuming no deterministic terms are given)
    delta_y_1_T : ndarray (neqs x nobs)
        :math:`(y_1, \\ldots, y_T) - (y_0, \\ldots, y_{T-1})`
    y_lag1 : ndarray (neqs x nobs)
        (dimensions assuming no deterministic terms are given)
        :math:`(y_0, \\ldots, y_{T-1})`

    Returns
    -------
    result : tuple
        A tuple of five ndarrays as well as eigenvalues and -vectors of a
        certain (matrix) product of some of the returned ndarrays.
        (See pp. 294-295 in [1]_ for more information on
        :math:`S_0, S_1, \\lambda_i, \\v_i` for
        :math:`i \\in \\{1, \\dots, K\\}`.)

    References
    ----------
    .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
    """
    nobs = y_lag1.shape[1]
    r0, r1 = _r_matrices(delta_y_1_T, y_lag1, delta_x)
    s00 = np.dot(r0, r0.T) / nobs
    s01 = np.dot(r0, r1.T) / nobs
    s10 = s01.T
    s11 = np.dot(r1, r1.T) / nobs
    s11_ = inv(_mat_sqrt(s11))
    s01_s11_ = np.dot(s01, s11_)
    eig = np.linalg.eig(s01_s11_.T @ inv(s00) @ s01_s11_)
    lambd = eig[0]
    v = eig[1]
    lambd_order = np.argsort(lambd)[::-1]
    lambd = lambd[lambd_order]
    v = v[:, lambd_order]
    return (s00, s01, s10, s11, s11_, lambd, v)