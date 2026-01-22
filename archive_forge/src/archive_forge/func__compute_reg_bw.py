import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, \
def _compute_reg_bw(self, bw):
    if not isinstance(bw, str):
        self._bw_method = 'user-specified'
        return np.asarray(bw)
    else:
        self._bw_method = bw
        if bw == 'cv_ls':
            res = self.cv_loo
        else:
            res = self.aic_hurvich
        X = np.std(self.exog, axis=0)
        h0 = 1.06 * X * self.nobs ** (-1.0 / (4 + np.size(self.exog, axis=1)))
        func = self.est[self.reg_type]
        bw_estimated = optimize.fmin(res, x0=h0, args=(func,), maxiter=1000.0, maxfun=1000.0, disp=0)
        return bw_estimated