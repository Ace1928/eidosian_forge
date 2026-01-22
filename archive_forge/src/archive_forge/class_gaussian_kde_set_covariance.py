import numpy as np
from numpy.testing import assert_almost_equal, assert_
import scipy
from scipy import stats
import matplotlib.pylab as plt
class gaussian_kde_set_covariance(stats.gaussian_kde):
    """
    from Anne Archibald in mailinglist:
    http://www.nabble.com/Width-of-the-gaussian-in-stats.kde.gaussian_kde---td19558924.html#a19558924
    """

    def __init__(self, dataset, covariance):
        self.covariance = covariance
        scipy.stats.gaussian_kde.__init__(self, dataset)

    def _compute_covariance(self):
        self.inv_cov = np.linalg.inv(self.covariance)
        self._norm_factor = np.sqrt(np.linalg.det(2 * np.pi * self.covariance)) * self.n