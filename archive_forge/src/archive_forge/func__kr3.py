from scipy import stats
import numpy as np
from statsmodels.tools.sm_exceptions import ValueWarning
def _kr3(y, alpha=5.0, beta=50.0):
    """
    KR3 estimator from Kim & White

    Parameters
    ----------
    y : array_like, 1-d
        Data to compute use in the estimator.
    alpha : float, optional
        Lower cut-off for measuring expectation in tail.
    beta :  float, optional
        Lower cut-off for measuring expectation in center.

    Returns
    -------
    kr3 : float
        Robust kurtosis estimator based on standardized lower- and upper-tail
        expected values

    Notes
    -----
    .. [*] Tae-Hwan Kim and Halbert White, "On more robust estimation of
       skewness and kurtosis," Finance Research Letters, vol. 1, pp. 56-73,
       March 2004.
    """
    perc = (alpha, 100.0 - alpha, beta, 100.0 - beta)
    lower_alpha, upper_alpha, lower_beta, upper_beta = np.percentile(y, perc)
    l_alpha = np.mean(y[y < lower_alpha])
    u_alpha = np.mean(y[y > upper_alpha])
    l_beta = np.mean(y[y < lower_beta])
    u_beta = np.mean(y[y > upper_beta])
    return (u_alpha - l_alpha) / (u_beta - l_beta)