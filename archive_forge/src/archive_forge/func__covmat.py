from statsmodels.compat.python import lzip
from statsmodels.compat.pandas import Appender
import numpy as np
from scipy import stats
import pandas as pd
import patsy
from collections import defaultdict
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults
from statsmodels.genmod import cov_struct as cov_structs
import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
import warnings
from statsmodels.graphics._regressionplots_doc import (
from statsmodels.discrete.discrete_margins import (
def _covmat(self):
    """
        Returns the sampling covariance matrix of the regression
        parameters and related quantities.

        Returns
        -------
        cov_robust : array_like
           The robust, or sandwich estimate of the covariance, which
           is meaningful even if the working covariance structure is
           incorrectly specified.
        cov_naive : array_like
           The model-based estimate of the covariance, which is
           meaningful if the covariance structure is correctly
           specified.
        cmat : array_like
           The center matrix of the sandwich expression, used in
           obtaining score test results.
        """
    endog = self.endog_li
    exog = self.exog_li
    weights = getattr(self, 'weights_li', None)
    varfunc = self.family.variance
    cached_means = self.cached_means
    bmat, cmat = (0, 0)
    for i in range(self.num_group):
        expval, lpr = cached_means[i]
        resid = endog[i] - expval
        dmat = self.mean_deriv(exog[i], lpr)
        sdev = np.sqrt(varfunc(expval))
        if weights is not None:
            w = weights[i]
            wresid = resid * w
            wdmat = dmat * w[:, None]
        else:
            wresid = resid
            wdmat = dmat
        rslt = self.cov_struct.covariance_matrix_solve(expval, i, sdev, (wdmat, wresid))
        if rslt is None:
            return (None, None, None, None)
        vinv_d, vinv_resid = tuple(rslt)
        bmat += np.dot(dmat.T, vinv_d)
        dvinv_resid = np.dot(dmat.T, vinv_resid)
        cmat += np.outer(dvinv_resid, dvinv_resid)
    scale = self.estimate_scale()
    try:
        bmati = np.linalg.inv(bmat)
    except np.linalg.LinAlgError:
        bmati = np.linalg.pinv(bmat)
    cov_naive = bmati * scale
    cov_robust = np.dot(bmati, np.dot(cmat, bmati))
    cov_naive *= self.scaling_factor
    cov_robust *= self.scaling_factor
    return (cov_robust, cov_naive, cmat)