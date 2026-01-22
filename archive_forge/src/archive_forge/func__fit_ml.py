import warnings
import numpy as np
from numpy.linalg import eigh, inv, norm, matrix_rank
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
from statsmodels.graphics.utils import _import_mpl
from .factor_rotation import rotate_factors, promax
def _fit_ml(self, start, em_iter, opt_method, opt):
    """estimate Factor model using Maximum Likelihood
        """
    if start is None:
        load, uniq = self._fit_ml_em(em_iter)
        start = self._pack(load, uniq)
    elif len(start) == 2:
        if len(start[1]) != start[0].shape[0]:
            msg = 'Starting values have incompatible dimensions'
            raise ValueError(msg)
        start = self._pack(start[0], start[1])
    else:
        raise ValueError('Invalid starting values')

    def nloglike(par):
        return -self.loglike(par)

    def nscore(par):
        return -self.score(par)
    if opt is None:
        opt = _opt_defaults
    r = minimize(nloglike, start, jac=nscore, method=opt_method, options=opt)
    if not r.success:
        warnings.warn('Fitting did not converge')
    par = r.x
    uniq, load = self._unpack(par)
    if uniq.min() < 1e-10:
        warnings.warn('Some uniquenesses are nearly zero')
    load = self._rotate(load, uniq)
    self.uniqueness = uniq
    self.communality = 1 - uniq
    self.loadings = load
    self.mle_retvals = r
    return FactorResults(self)