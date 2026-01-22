import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
def _compute_S(self, D, sigma):
    """covariance of observations (nobs_i, nobs_i)  (JP check)
        Display (3.3) from Laird, Lange, Stram (see help(Unit))
        """
    self.S = np.identity(self.n) * sigma ** 2 + np.dot(self.Z, np.dot(D, self.Z.T))