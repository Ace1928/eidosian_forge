from pandas
import numpy as np
from statsmodels.regression.linear_model import GLS, RegressionResults
@property
def wrnorm_cov_params(self):
    """
        Heteroskedasticity-consistent parameter covariance
        Used to calculate White standard errors.
        """
    if self._wncp is None:
        df = self.df_resid
        pred = np.dot(self.wexog, self.coeffs)
        eps = np.diag((self.wendog - pred) ** 2)
        sigmaSq = np.sum(eps)
        pinvX = np.dot(self.rnorm_cov_params, self.wexog.T)
        self._wncp = np.dot(np.dot(pinvX, eps), pinvX.T) * df / sigmaSq
    return self._wncp