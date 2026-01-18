import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
def var_ddof(self, ddof=0):
    """variance of data given ddof

        Parameters
        ----------
        ddof : int, float
            degrees of freedom correction, independent of attribute ddof

        Returns
        -------
        var : float, ndarray
            variance with denominator ``sum_weights - ddof``
        """
    return self.sumsquares / (self.sum_weights - ddof)