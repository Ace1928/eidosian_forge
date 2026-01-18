import numpy as np
import pandas as pd
from statsmodels.base.model import LikelihoodModelResults
def update_mean(self):
    """
        Gibbs update of the mean vector.

        Do not call until update_data has been called once.
        """
    cm = np.linalg.solve(self.cov / self.nobs + self.mean_prior, self.mean_prior / self.nobs)
    cm = np.dot(self.cov, cm)
    vm = np.linalg.solve(self.cov, self._data.sum(0))
    vm = np.dot(cm, vm)
    r = np.linalg.cholesky(cm)
    self.mean = vm + np.dot(r, np.random.normal(0, 1, self.nvar))