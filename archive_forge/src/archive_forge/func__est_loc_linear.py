import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, \
def _est_loc_linear(self, bw, endog, exog, data_predict, W):
    """
        Local linear estimator of g(x) in the regression ``y = g(x) + e``.

        Parameters
        ----------
        bw : array_like
            Vector of bandwidth value(s)
        endog : 1D array_like
            The dependent variable
        exog : 1D or 2D array_like
            The independent variable(s)
        data_predict : 1D array_like of length K, where K is
            the number of variables. The point at which
            the density is estimated

        Returns
        -------
        D_x : array_like
            The value of the conditional mean at data_predict

        Notes
        -----
        See p. 81 in [1] and p.38 in [2] for the formulas
        Unlike other methods, this one requires that data_predict be 1D
        """
    nobs, k_vars = exog.shape
    ker = gpke(bw, data=exog, data_predict=data_predict, var_type=self.var_type, ckertype=self.ckertype, ukertype=self.ukertype, okertype=self.okertype, tosum=False)
    ker = W * ker[:, np.newaxis]
    M12 = exog - data_predict
    M22 = np.dot(M12.T, M12 * ker)
    M12 = (M12 * ker).sum(axis=0)
    M = np.empty((k_vars + 1, k_vars + 1))
    M[0, 0] = ker.sum()
    M[0, 1:] = M12
    M[1:, 0] = M12
    M[1:, 1:] = M22
    ker_endog = ker * endog
    V = np.empty((k_vars + 1, 1))
    V[0, 0] = ker_endog.sum()
    V[1:, 0] = ((exog - data_predict) * ker_endog).sum(axis=0)
    mean_mfx = np.dot(np.linalg.pinv(M), V)
    mean = mean_mfx[0]
    mfx = mean_mfx[1:, :]
    return (mean, mfx)