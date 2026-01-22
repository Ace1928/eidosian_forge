import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
def _compute_D(self, ML=False):
    """
        Estimate random effects covariance D.
        If ML is True, return the ML estimate of sigma,
        else return the REML estimate.

        If ML, this is (3.7) in Laird, Lange, Stram (see help(Mixed)),
        otherwise it corresponds to (3.9).
        """
    D = 0.0
    for unit in self.units:
        if ML:
            W = unit.W
        else:
            unit.compute_P(self.Sinv)
            W = unit.P
        D += np.multiply.outer(unit.b, unit.b)
        t = np.dot(unit.Z, self.D)
        D += self.D - np.dot(np.dot(t.T, W), t)
    self.D = D / self.m