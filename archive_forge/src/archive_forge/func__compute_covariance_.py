import numpy as np
from numpy.testing import assert_almost_equal, assert_
import scipy
from scipy import stats
import matplotlib.pylab as plt
def _compute_covariance_(self):
    """not used"""
    self.inv_cov = np.linalg.inv(self.covariance)
    self._norm_factor = np.sqrt(np.linalg.det(2 * np.pi * self.covariance)) * self.n