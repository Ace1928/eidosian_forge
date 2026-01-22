import numpy as np
from statsmodels.stats._knockoff import RegressionFDR
def _ecdf(x):
    """no frills empirical cdf used in fdrcorrection
    """
    nobs = len(x)
    return np.arange(1, nobs + 1) / float(nobs)