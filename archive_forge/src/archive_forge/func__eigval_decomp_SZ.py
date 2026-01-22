import numpy as np
import numpy.linalg as la
import scipy.linalg as L
from statsmodels.tools.decorators import cache_readonly
import statsmodels.tsa.tsatools as tsa
import statsmodels.tsa.vector_ar.plotting as plotting
import statsmodels.tsa.vector_ar.util as util
def _eigval_decomp_SZ(self, irf_resim):
    """
        Returns
        -------
        W: array of eigenvectors
        eigva: list of eigenvalues
        k: matrix indicating column # of largest eigenvalue for each c_i,j
        """
    neqs = self.neqs
    periods = self.periods
    cov_hold = np.zeros((neqs, neqs, periods, periods))
    for i in range(neqs):
        for j in range(neqs):
            cov_hold[i, j, :, :] = np.cov(irf_resim[:, 1:, i, j], rowvar=0)
    W = np.zeros((neqs, neqs, periods, periods))
    eigva = np.zeros((neqs, neqs, periods, 1))
    k = np.zeros((neqs, neqs), dtype=int)
    for i in range(neqs):
        for j in range(neqs):
            W[i, j, :, :], eigva[i, j, :, 0], k[i, j] = util.eigval_decomp(cov_hold[i, j, :, :])
    return (W, eigva, k)