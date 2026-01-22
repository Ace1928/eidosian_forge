import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def DescStat(endog):
    """
    Returns an instance to conduct inference on descriptive statistics
    via empirical likelihood.  See DescStatUV and DescStatMV for more
    information.

    Parameters
    ----------
    endog : ndarray
         Array of data

    Returns : DescStat instance
        If k=1, the function returns a univariate instance, DescStatUV.
        If k>1, the function returns a multivariate instance, DescStatMV.
    """
    if endog.ndim == 1:
        endog = endog.reshape(len(endog), 1)
    if endog.shape[1] == 1:
        return DescStatUV(endog)
    if endog.shape[1] > 1:
        return DescStatMV(endog)