from statsmodels.base.elastic_net import RegularizedResults
from statsmodels.stats.regularized_covariance import _calc_nodewise_row, \
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
import numpy as np
def _join_naive(params_l, threshold=0):
    """joins the results from each run of _est_<type>_naive
    and returns the mean estimate of the coefficients

    Parameters
    ----------
    params_l : list
        A list of arrays of coefficients.
    threshold : scalar
        The threshold at which the coefficients will be cut.
    """
    p = len(params_l[0])
    partitions = len(params_l)
    params_mn = np.zeros(p)
    for params in params_l:
        params_mn += params
    params_mn /= partitions
    params_mn[np.abs(params_mn) < threshold] = 0
    return params_mn