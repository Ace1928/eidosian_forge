from statsmodels.base.elastic_net import RegularizedResults
from statsmodels.stats.regularized_covariance import _calc_nodewise_row, \
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
import numpy as np
def _join_debiased(results_l, threshold=0):
    """joins the results from each run of _est_regularized_debiased
    and returns the debiased estimate of the coefficients

    Parameters
    ----------
    results_l : list
        A list of tuples each one containing the params, grad,
        nodewise_row and nodewise_weight values for each partition.
    threshold : scalar
        The threshold at which the coefficients will be cut.
    """
    p = len(results_l[0][0])
    partitions = len(results_l)
    params_mn = np.zeros(p)
    grad_mn = np.zeros(p)
    nodewise_row_l = []
    nodewise_weight_l = []
    for r in results_l:
        params_mn += r[0]
        grad_mn += r[1]
        nodewise_row_l.extend(r[2])
        nodewise_weight_l.extend(r[3])
    nodewise_row_l = np.array(nodewise_row_l)
    nodewise_weight_l = np.array(nodewise_weight_l)
    params_mn /= partitions
    grad_mn *= -1.0 / partitions
    approx_inv_cov = _calc_approx_inv_cov(nodewise_row_l, nodewise_weight_l)
    debiased_params = params_mn + approx_inv_cov.dot(grad_mn)
    debiased_params[np.abs(debiased_params) < threshold] = 0
    return debiased_params