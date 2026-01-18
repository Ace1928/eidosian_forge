import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as student_t
from scipy import stats
from statsmodels.tools.tools import clean0, fullrank
from statsmodels.stats.multitest import multipletests
def wald_test_noncent_generic(params, r_matrix, value, cov_params, diff=None, joint=True):
    """noncentrality parameter for a wald test

    The null hypothesis is ``diff = r_matrix @ params - value = 0``

    Parameters
    ----------
    params : ndarray
        parameters of the model at which to evaluate noncentrality. This can
        be estimated parameters or parameters under an alternative.
    r_matrix : ndarray
        Restriction matrix or contrasts for the Null hypothesis

    value : None or ndarray
        Value of the linear combination of parameters under the null
        hypothesis. If value is None, then it will be replace by zero.
    cov_params : ndarray
        covariance matrix of the parameter estimates
    diff : None or ndarray
        If diff is not None, then it will be used instead of
        ``diff = r_matrix @ params - value``
    joint : bool
        If joint is True, then the noncentrality parameter for the joint
        hypothesis will be returned.
        If joint is True, then an array of noncentrality parameters will be
        returned, where elements correspond to rows of the restriction matrix.
        This correspond to the `t_test` in models and is not a quadratic form.

    Returns
    -------
    nc : float or ndarray
        Noncentrality parameter for Wald tests, correspondig to `wald_test`
        or `t_test` depending on whether `joint` is true or not.
        It needs to be divided by nobs to obtain effect size.


    Notes
    -----
    Status : experimental, API will likely change
    """
    if value is None:
        value = 0
    if diff is None:
        diff = r_matrix @ params - value
    c = r_matrix
    cov_c = c.dot(cov_params).dot(c.T)
    if joint:
        nc = diff @ np.linalg.solve(cov_c, diff)
    else:
        nc = diff / np.sqrt(np.diag(cov_c))
    return nc