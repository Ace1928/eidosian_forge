import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from . import kernels
def _cv_ls(self):
    """
        Returns the cross-validation least squares bandwidth parameter(s).

        Notes
        -----
        For more details see pp. 16, 27 in Ref. [1] (see module docstring).

        Returns the value of the bandwidth that maximizes the integrated mean
        square error between the estimated and actual distribution.  The
        integrated mean square error (IMSE) is given by:

        .. math:: \\int\\left[\\hat{f}(x)-f(x)\\right]^{2}dx

        This is the general formula for the IMSE.  The IMSE differs for
        conditional (``KDEMultivariateConditional``) and unconditional
        (``KDEMultivariate``) kernel density estimation.
        """
    h0 = self._normal_reference()
    bw = optimize.fmin(self.imse, x0=h0, maxiter=1000.0, maxfun=1000.0, disp=0, xtol=0.001)
    bw = self._set_bw_bounds(bw)
    return bw