import numpy as np
import pandas as pd
from statsmodels.base.model import LikelihoodModelResults
def update_cov(self):
    """
        Gibbs update of the covariance matrix.

        Do not call until update_data has been called once.
        """
    r = self._data - self.mean
    gr = np.dot(r.T, r)
    a = gr + self.cov_prior
    df = int(np.ceil(self.nobs + self.cov_prior_df))
    r = np.linalg.cholesky(np.linalg.inv(a))
    x = np.dot(np.random.normal(size=(df, self.nvar)), r.T)
    ma = np.dot(x.T, x)
    self.cov = np.linalg.inv(ma)